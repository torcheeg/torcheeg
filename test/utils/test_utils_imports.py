import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestUtilsImports(unittest.TestCase):
    def run_fresh(self, body):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_utilities_do_not_load_visualizers(self):
        self.run_fresh('''
            import importlib.abc
            import sys
            import torcheeg

            class BlockVisualizers(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith('torcheeg.visualizers'):
                        raise AssertionError('Unexpected visualization import: ' + fullname)

            sys.meta_path.insert(0, BlockVisualizers())
            import torcheeg.utils as utils
            from torcheeg.utils import download_url, get_random_dir_path, LazyLoader
            assert download_url is utils.download_url
            assert not any(name.startswith('torcheeg.visualizers') for name in sys.modules)
        ''')


if __name__ == '__main__':
    unittest.main()
