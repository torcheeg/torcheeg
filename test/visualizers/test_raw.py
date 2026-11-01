import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from torcheeg.datasets import DEAPDataset
from torcheeg.visualizers import plot_signal


class TestRaw(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

    def test_plot_signal(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_signal(
            mock_eeg, channel_list=DEAPDataset.channels, sampling_rate=128)

        self.assertEqual(img.ndim, 3)
        self.assertGreater(np.ptp(img[..., :3]), 0)


if __name__ == '__main__':
    unittest.main()
