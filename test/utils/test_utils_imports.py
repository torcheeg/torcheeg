import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class TestUtilsImports(unittest.TestCase):
    def run_fresh(self, body):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lightweight_utilities_do_not_load_plotting_or_experiment(self):
        self.run_fresh('''
            import importlib.abc
            import sys
            import torcheeg

            blocked = {'torcheeg.utils.visualize', 'torcheeg.utils.experiment'}
            assert not blocked.intersection(sys.modules)

            class BlockHeavyUtilities(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in blocked:
                        raise AssertionError('Unexpected utility import: ' + fullname)

            sys.meta_path.insert(0, BlockHeavyUtilities())
            import torcheeg.utils as utils
            from torcheeg.utils import download_url, get_random_dir_path
            from torcheeg.utils.file import download_url as direct_download
            from torcheeg.utils.lazy_loader import LazyLoader
            assert download_url is direct_download
            assert utils.download_url is direct_download
            assert 'plot_graph' in dir(utils)
            assert not blocked.intersection(sys.modules)
        ''')

    def test_public_exports_resolve_to_original_functions(self):
        self.run_fresh('''
            import torcheeg.utils as utils
            from torcheeg.utils import experiment, file, visualize

            expected = {
                'seed_everything': experiment,
                **{name: file for name in (
                    'download_url', 'get_package_dir_path',
                    'get_random_dir_path', 'get_temp_dir_path')},
                **{name: visualize for name in (
                    'plot_2d_tensor', 'plot_3d_tensor', 'plot_adj_connectivity',
                    'plot_feature_topomap', 'plot_graph', 'plot_raw_topomap',
                    'plot_signal')},
            }
            assert set(utils.__all__) == set(expected)
            namespace = {}
            exec('from torcheeg.utils import *', namespace)
            for name, module in expected.items():
                value = getattr(utils, name)
                assert value is getattr(module, name)
                assert utils.__dict__[name] is value
                assert namespace[name] is value
            try:
                utils.nonexistent_utility
            except AttributeError:
                pass
            else:
                raise AssertionError('Unknown attributes must raise AttributeError')
        ''')


if __name__ == '__main__':
    unittest.main()
