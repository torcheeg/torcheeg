import unittest

import torch

from torch_geometric.data import Data, Batch
from torcheeg.models import DGCNN, RGNN


class TestGNN(unittest.TestCase):
    def test_dgcnn(self):
        eeg = torch.randn(1, 62, 200)
        model = DGCNN(in_channels=200, num_electrodes=62, hid_channels=32, num_layers=2, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))

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
                     domain_adaptation=False,
                     alpha=0.0,
                     learn_edge_weights=True)
        pred = model(data)

        self.assertEqual(tuple(pred.shape), (2, 2))

        data = data.cuda()
        model = model.cuda()
        pred = model(data)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
