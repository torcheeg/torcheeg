import unittest

import numpy as np
import torch

from torcheeg.datasets import DEAPDataset, SEEDDataset
from torcheeg.datasets.montage import GENERAL_REGION_LIST
from torcheeg.datasets.utils import get_adjacency, get_locations, get_regions
from torcheeg.transforms.graph import ToG
from torcheeg.visualizers import (plot_2d_tensor, plot_3d_tensor,
                                  plot_adj_connectivity, plot_feature_topomap,
                                  plot_graph, plot_raw_topomap, plot_signal)


class TestVisualize(unittest.TestCase):

    def test_plot_raw_topomap(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_raw_topomap(
            mock_eeg, channel_list=DEAPDataset.channels, sampling_rate=128)
        # not all zero (numpy array)
        self.assertFalse(np.all(img == 0))

    def test_plot_feature_topomap(self):
        mock_eeg = torch.randn(32, 4)
        img = plot_feature_topomap(mock_eeg,
                                   channel_list=DEAPDataset.channels,
                                   feature_list=["theta", "alpha", "beta", "gamma"])

        self.assertFalse(np.all(img == 0))
        img = plot_feature_topomap(mock_eeg,
                                   channel_list=DEAPDataset.channels,
                                   feature_list=["theta", "alpha", "beta", "gamma"], fig_shape=(2, 2))

        self.assertFalse(np.all(img == 0))

    def test_plot_signal(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_signal(
            mock_eeg, channel_list=DEAPDataset.channels, sampling_rate=128)

        self.assertFalse(np.all(img == 0))

    def test_plot_3d_tensor(self):
        mock_eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(mock_eeg)

        self.assertFalse(np.all(img == 0))

    def test_plot_2d_tensor(self):
        mock_eeg = torch.randn(9, 9)
        img = plot_2d_tensor(mock_eeg)

        self.assertFalse(np.all(img == 0))

    def test_plot_graph(self):
        mock_eeg = np.random.randn(32, 128)
        mock_g = ToG(get_adjacency(DEAPDataset.channels,
                     DEAPDataset.adjacency))(eeg=mock_eeg)['eeg']
        img = plot_graph(mock_g, get_locations(
            DEAPDataset.channels, DEAPDataset.grid))

        self.assertFalse(np.all(img == 0))

    def test_plot_adj_connectivity(self):
        mock_adj = torch.randn(62, 62)
        img = plot_adj_connectivity(mock_adj,
                                    SEEDDataset.channels,
                                    get_regions(SEEDDataset.channels,
                                                GENERAL_REGION_LIST),
                                    num_connectivity=60,
                                    linewidth=1.5)

        self.assertFalse(np.all(img == 0))


if __name__ == '__main__':
    unittest.main()
