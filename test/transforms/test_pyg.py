import unittest

import numpy as np
import torch
from torch_geometric.data import Data
from torcheeg.datasets.constants import DEAP_ADJACENCY_MATRIX
from torcheeg.transforms.pyg import ToDynamicG, ToG


class TestPyGTransforms(unittest.TestCase):
    def test_to_g(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToG(DEAP_ADJACENCY_MATRIX)(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

        eeg = torch.randn(32, 128)
        transformed_eeg = ToG(DEAP_ADJACENCY_MATRIX)(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

    def test_to_dynamic_g(self):
        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func='gaussian_distance', sigma=1.0, top_k=10, complete_graph=False)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

        eeg = torch.randn(32, 128)
        transform = ToDynamicG(edge_func='gaussian_distance', sigma=1.0, top_k=10, complete_graph=False)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func='absolute_pearson_correlation_coefficient', threshold=0.1, binary=True)
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func='phase_locking_value')
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))

        eeg = np.random.randn(32, 128)
        transform = ToDynamicG(edge_func=lambda x, y: (x * y).mean())
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(isinstance(transformed_eeg, Data))


if __name__ == '__main__':
    unittest.main()
