import unittest

import numpy as np
from torch_geometric.data import Data
from torcheeg.datasets.constants import DEAP_ADJACENCY_MATRIX
from torcheeg.transforms import ToG


class TestPyGTransforms(unittest.TestCase):
    def test_to_tensor(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToG(DEAP_ADJACENCY_MATRIX)(eeg)
        self.assertTrue(isinstance(transformed_eeg, Data))


if __name__ == '__main__':
    unittest.main()
