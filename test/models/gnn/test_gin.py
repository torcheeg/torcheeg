import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from torcheeg.models import GIN


class TestOptionalModelDependencies(unittest.TestCase):
    def _run_without(self, modules, body):
        root = Path(__file__).resolve().parents[3]
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

    def test_gin_reports_missing_pyg_when_constructed(self):
        self._run_without(('torch_geometric',), '''
            from torcheeg.models import GIN
            for build in (GIN,):
                try:
                    build()
                except ImportError as error:
                    assert 'Could not import optional dependency `torch_geometric`' in str(error), str(error)
                else:
                    raise AssertionError('Graph model must report its missing dependency')
        ''')


class TestGIN(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        adj = torch.rand(62, 62)
        adj = (adj > 0.5).float()
        adj[adj == 0] = 1e-6  # to construct complete graph
        sparse_adj = adj.to_sparse()

        data = Data(edge_index=sparse_adj._indices())
        data.x = torch.randn(62, 4)
        data.edge_weight = sparse_adj._values()
        data = Batch.from_data_list([data, data])

        model = GIN(in_channels=4, hid_channels=64, num_classes=2)
        pred = model(data)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())

        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
