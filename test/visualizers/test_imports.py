import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class TestVisualizerImports(unittest.TestCase):
    def run_fresh(self, body):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tensor_import_does_not_load_other_visualizers(self):
        self.run_fresh('''
            import importlib.abc
            import sys
            import torcheeg
            blocked = {'torcheeg.visualizers.raw', 'torcheeg.visualizers.topomap',
                       'torcheeg.visualizers.connectivity', 'torcheeg.visualizers.graph'}

            class BlockOtherVisualizers(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in blocked:
                        raise AssertionError('Unexpected visualization import: ' + fullname)

            sys.meta_path.insert(0, BlockOtherVisualizers())
            from torcheeg.visualizers import plot_2d_tensor
            from torcheeg.visualizers.tensor import plot_2d_tensor as direct
            assert direct is plot_2d_tensor
            assert not blocked.intersection(sys.modules)
        ''')

    def test_public_exports(self):
        self.run_fresh('''
            import torcheeg.visualizers as visualizers
            from torcheeg.visualizers import raw, topomap, tensor, connectivity, graph
            expected = {
                'plot_signal': raw,
                'plot_raw_topomap': topomap,
                'plot_feature_topomap': topomap,
                'plot_2d_tensor': tensor,
                'plot_3d_tensor': tensor,
                'plot_adj_connectivity': connectivity,
                'plot_graph': graph,
            }
            assert set(visualizers.__all__) == set(expected)
            namespace = {}
            exec('from torcheeg.visualizers import *', namespace)
            for name, module in expected.items():
                value = getattr(visualizers, name)
                assert value is getattr(module, name)
                assert visualizers.__dict__[name] is value
                assert namespace[name] is value
            try:
                visualizers.nonexistent_plot
            except AttributeError:
                pass
            else:
                raise AssertionError('Unknown attributes must raise AttributeError')
        ''')


if __name__ == '__main__':
    unittest.main()
