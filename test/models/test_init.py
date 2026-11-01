import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestOptionalModelDependencies(unittest.TestCase):
    def run_without(self, modules, body):
        root = Path(__file__).resolve().parents[2]
        script = '''
import importlib.abc
import sys
import torch
import torcheeg

assert torcheeg.__file__.startswith(ROOT)

blocked = MODULES

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)

assert not set(blocked).intersection(sys.modules)
sys.meta_path.insert(0, BlockOptionalDependencies())
'''.replace('MODULES', repr(modules)).replace('ROOT', repr(str(root))) + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_non_graph_models_without_graph_plugins(self):
        for modules in [('torch_scatter',), ('torch_geometric',), ('torch_sparse',),
                        ('torch_scatter', 'torch_geometric', 'torch_sparse')]:
            with self.subTest(modules=modules):
                self.run_without(modules, '''
                    from torcheeg.models import CCNN, GRU, VanillaTransformer
                    cases = [
                        (CCNN(in_channels=4, grid_size=(9, 9), num_classes=2), (2, 4, 9, 9)),
                        (GRU(num_electrodes=3, hid_channels=4, num_classes=2), (2, 3, 16)),
                        (VanillaTransformer(num_electrodes=3, chunk_size=16,
                            t_patch_size=4, hid_channels=8, depth=1, heads=2,
                            head_channels=4, mlp_channels=8, num_classes=2), (2, 3, 16))]
                    for model, shape in cases:
                        x = torch.randn(*shape, requires_grad=True)
                        out = model(x)
                        assert out.shape == (2, 2)
                        out.square().mean().backward()
                        assert torch.isfinite(x.grad).all()
                        assert x.grad.abs().sum() > 0
                    assert not set(blocked).intersection(sys.modules)
                ''')

    def test_graph_entry_points_share_loader_without_importing_pyg(self):
        self.run_without(('torch_geometric',), """
            from torcheeg.models.gnn import gin, rgnn
            from torcheeg.transforms.graph import to
            from torcheeg.visualizers import graph
            from torcheeg.utils import LazyLoader
            for loader in (gin._pyg, rgnn._pyg, to.pyg, graph._pyg):
                assert type(loader) is LazyLoader
                assert loader._module is None
            transform = to.ToG(adj=torch.eye(3))
            try:
                transform(eeg=torch.ones(3, 4))
            except ImportError as error:
                assert 'Could not import optional dependency `torch_geometric`' in str(error), str(error)
            else:
                raise AssertionError('Graph conversion must load its dependency')
        """)


if __name__ == '__main__':
    unittest.main()
