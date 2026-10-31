import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from torcheeg.models import RGNN


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
