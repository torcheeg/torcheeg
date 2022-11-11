import unittest

import numpy as np
import torch
from torcheeg.datasets.constants.emotion_recognition.deap import (DEAP_ADJACENCY_MATRIX, DEAP_CHANNEL_LIST,
                                                                  DEAP_CHANNEL_LOCATION_DICT, DEAP_LOCATION_LIST)
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LIST, SEED_GENERAL_REGION_LIST
from torcheeg.transforms.pyg import ToG
from torcheeg.utils import (plot_2d_tensor, plot_3d_tensor, plot_feature_topomap, plot_raw_topomap, plot_signal,
                            plot_adj_connectivity)
from torcheeg.utils.pyg import plot_graph


class TestVisualize(unittest.TestCase):

    def test_plot_raw_topomap(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_raw_topomap(mock_eeg, channel_list=DEAP_CHANNEL_LIST, sampling_rate=128)
        self.assertEqual(img.shape, (347, 1550, 4))

    def test_plot_feature_topomap(self):
        mock_eeg = torch.randn(32, 4)
        img = plot_feature_topomap(mock_eeg,
                                   channel_list=DEAP_CHANNEL_LIST,
                                   feature_list=["theta", "alpha", "beta", "gamma"])
        self.assertEqual(img.shape, (347, 1550, 4))

    def test_plot_signal(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_signal(mock_eeg, channel_list=DEAP_CHANNEL_LIST, sampling_rate=128)
        self.assertEqual(img.shape, (773, 727, 4))

    def test_plot_3d_tensor(self):
        mock_eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(mock_eeg)
        self.assertEqual(img.shape, (374, 382, 4))

    def test_plot_2d_tensor(self):
        mock_eeg = torch.randn(9, 9)
        img = plot_2d_tensor(mock_eeg)
        self.assertEqual(img.shape, (393, 388, 4))

    def test_plot_graph(self):
        mock_eeg = np.random.randn(32, 128)
        mock_g = ToG(DEAP_ADJACENCY_MATRIX)(eeg=mock_eeg)['eeg']
        img = plot_graph(mock_g, DEAP_CHANNEL_LOCATION_DICT)
        self.assertEqual(img.shape, (494, 599, 4))

    def test_plot_adj_connectivity(self):
        mock_adj = torch.randn(62, 62)
        img = plot_adj_connectivity(mock_adj,
                                    SEED_CHANNEL_LIST,
                                    SEED_GENERAL_REGION_LIST,
                                    num_connectivity=60,
                                    linewidth=1.5)
        self.assertEqual(img.shape, (771, 773, 4))


if __name__ == '__main__':
    unittest.main()
