import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from torcheeg.datasets import DEAPDataset
from torcheeg.visualizers import plot_feature_topomap, plot_raw_topomap


class TestTopomap(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

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


if __name__ == '__main__':
    unittest.main()
