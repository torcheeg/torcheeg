import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from torcheeg.models import RGNN


class TestOptionalModelDependencies(unittest.TestCase):
    def run_without(self, modules, body):
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

    def test_rgnn_reports_missing_scatter_when_constructed(self):
        self.run_without(('torch_scatter',), '''
            from torcheeg.models import RGNN
            try:
                RGNN(torch.eye(3), num_electrodes=3)
            except ImportError as error:
                assert 'Could not import optional dependency `torch_scatter`' in str(error), str(error)
                assert 'pytorch-geometric.readthedocs.io' in str(error), str(error)
            else:
                raise AssertionError('RGNN must report its missing dependency')
        ''')


class TestRgnn(unittest.TestCase):

    def test_rgnn(self):
        adj = torch.rand(62, 62)
        adj = (adj > 0.5).float()
        adj[adj == 0] = 1e-6  # to construct complete graph
        sparse_adj = adj.to_sparse()

        data = Data(edge_index=sparse_adj._indices())
        data.x = torch.randn(62, 200)
        data.edge_weight = sparse_adj._values()
        data = Batch.from_data_list([data, data])

        model = RGNN(adj=adj,
                     in_channels=200,
                     num_electrodes=62,
                     hid_channels=32,
                     num_layers=2,
                     num_classes=2,
                     dropout=0.7,
                     learn_edge_weights=True)
        pred = model(data)

        self.assertEqual(tuple(pred.shape), (2, 2))


class TestRGNNNormalization(unittest.TestCase):
    def test_signed_graph_outputs_and_gradients_match_dense_reference(self):
        torch.manual_seed(42)
        adj = torch.tensor([[1., -.4, .2], [-.4, .5, -.7], [.2, -.7, 1.2]])
        model = RGNN(adj, num_electrodes=3, in_channels=2, hid_channels=3,
                     num_layers=2, num_classes=2, dropout=0.).eval()
        x = torch.randn(2, 3, 2, requires_grad=True)
        edge_index = torch.ones(3, 3).nonzero().t().contiguous()
        batch = Batch.from_data_list(
            [Data(x=row, edge_index=edge_index) for row in x])
        actual = model(batch)
        actual.square().sum().backward()

        parameters = {name: parameter.detach().clone().requires_grad_()
                      for name, parameter in model.named_parameters()}
        y = x.detach().clone().requires_grad_()
        lower = torch.zeros(3, 3)
        lower[model.xs, model.ys] = parameters['adj']
        weights = lower + lower.t() - torch.diag(lower.diagonal())
        inv_degree = weights.abs().sum(dim=1).rsqrt()
        normalized = inv_degree[:, None] * weights * inv_degree[None, :]
        features = y
        for _ in range(2):
            features = normalized.t() @ features
        features = F.relu(F.linear(features, parameters['conv1.lin.weight'],
                                   parameters['conv1.lin.bias']))
        expected = F.linear(features.sum(
            dim=1), parameters['fc.weight'], parameters['fc.bias'])
        expected.square().sum().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, parameters[name].grad,
                                       rtol=1e-5, atol=1e-6, msg=name)
        self.assertGreater(model.adj.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
