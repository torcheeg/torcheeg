import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_adjacency
from torcheeg.transforms import ToDynamicG, ToG


class TestPyGTransforms(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_to_g(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToG(get_adjacency(
            DEAPDataset.channels, DEAPDataset.adjacency))(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

        eeg = torch.randn(32, 128)
        transformed_eeg = ToG(get_adjacency(
            DEAPDataset.channels, DEAPDataset.adjacency))(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

    def test_to_dynamic_g(self):
        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func='gaussian_distance',
                               sigma=1.0, top_k=10, complete_graph=False)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

        eeg = torch.randn(32, 128)
        transform = ToDynamicG(edge_func='gaussian_distance',
                               sigma=1.0, top_k=10, complete_graph=False)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(
            edge_func='absolute_pearson_correlation_coefficient', threshold=0.1, binary=True)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func='phase_locking_value')
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func=lambda x, y: (x * y).mean())
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertIsInstance(transformed_eeg, Data)
        torch.testing.assert_close(
            transformed_eeg.x, torch.as_tensor(eeg).float())
        self.assertEqual(transformed_eeg.edge_index.shape[0], 2)
        self.assertEqual(transformed_eeg.edge_weight.numel(),
                         transformed_eeg.edge_index.shape[1])
        self.assertTrue(torch.isfinite(transformed_eeg.edge_weight).all())


if __name__ == '__main__':
    unittest.main()
