import unittest

import matplotlib.pyplot as plt
import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_adjacency, get_locations
from torcheeg.transforms.graph import ToG
from torcheeg.visualizers import plot_graph


class TestGraph(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

    def test_plot_graph(self):
        mock_eeg = np.random.randn(32, 128)
        mock_g = ToG(get_adjacency(DEAPDataset.channels,
                     DEAPDataset.adjacency))(mock_eeg)
        img = plot_graph(mock_g, get_locations(
            DEAPDataset.channels, DEAPDataset.grid))
        self.assertEqual(img.ndim, 3)
        self.assertGreater(np.ptp(img[..., :3]), 0)


if __name__ == '__main__':
    unittest.main()
