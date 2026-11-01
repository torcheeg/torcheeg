import unittest

import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_adjacency, get_locations
from torcheeg.transforms.graph import ToG
from torcheeg.visualizers import plot_graph


class TestGraph(unittest.TestCase):

    def test_plot_graph(self):
        mock_eeg = np.random.randn(32, 128)
        mock_g = ToG(get_adjacency(DEAPDataset.channels,
                     DEAPDataset.adjacency))(eeg=mock_eeg)['eeg']
        img = plot_graph(mock_g, get_locations(
            DEAPDataset.channels, DEAPDataset.grid))

        self.assertFalse(np.all(img == 0))


if __name__ == '__main__':
    unittest.main()
