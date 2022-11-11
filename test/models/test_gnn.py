import unittest

import torch
from torch_geometric.data import Batch, Data

from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_GENERAL_REGION_LIST
from torcheeg.models import DGCNN, LGGNet
from torcheeg.models.pyg import GIN, RGNN


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

    def test_lggnet(self):
        eeg = torch.rand(2, 1, 32, 128)
        model = LGGNet(DEAP_GENERAL_REGION_LIST, num_electrodes=32, chunk_size=128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (2, 2))

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

        data = data.cuda()
        model = model.cuda()
        pred = model(data)
        self.assertEqual(tuple(pred.shape), (2, 2))

    def test_gin(self):
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

        self.assertEqual(tuple(pred.shape), (2, 2))

        data = data.cuda()
        model = model.cuda()
        pred = model(data)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
