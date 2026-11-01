import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestVisualizerImports(unittest.TestCase):

    def _run_fresh(self, body):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(body)], cwd=root, env=dict(
            os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tensor_import_does_not_load_other_visualizers(self):
        self._run_fresh("\n            import importlib.abc\n            import sys\n            import torcheeg\n            blocked = {'torcheeg.visualizers.raw', 'torcheeg.visualizers.topomap',\n                       'torcheeg.visualizers.connectivity', 'torcheeg.visualizers.graph'}\n\n            class BlockOtherVisualizers(importlib.abc.MetaPathFinder):\n                def find_spec(self, fullname, path=None, target=None):\n                    if fullname in blocked:\n                        raise AssertionError('Unexpected visualization import: ' + fullname)\n\n            sys.meta_path.insert(0, BlockOtherVisualizers())\n            from torcheeg.visualizers import plot_2d_tensor\n            from torcheeg.visualizers.tensor import plot_2d_tensor as direct\n            assert direct is plot_2d_tensor\n            assert not blocked.intersection(sys.modules)\n        ")

    def test_public_exports(self):
        self._run_fresh(
            "\n            import torcheeg.visualizers as visualizers\n            from torcheeg.visualizers import raw, topomap, tensor, connectivity, graph\n            expected = {\n                'plot_signal': raw,\n                'plot_raw_topomap': topomap,\n                'plot_feature_topomap': topomap,\n                'plot_2d_tensor': tensor,\n                'plot_3d_tensor': tensor,\n                'plot_adj_connectivity': connectivity,\n                'plot_graph': graph,\n            }\n            assert set(visualizers.__all__) == set(expected)\n            namespace = {}\n            exec('from torcheeg.visualizers import *', namespace)\n            for name, module in expected.items():\n                value = getattr(visualizers, name)\n                assert value is getattr(module, name)\n                assert visualizers.__dict__[name] is value\n                assert namespace[name] is value\n            try:\n                visualizers.nonexistent_plot\n            except AttributeError:\n                pass\n            else:\n                raise AssertionError('Unknown attributes must raise AttributeError')\n        ")


if __name__ == '__main__':
    unittest.main()
