import unittest

import numpy as np
import torch

from torcheeg.datasets import DEAPDataset
from torcheeg.visualizers import plot_signal


class TestRaw(unittest.TestCase):

    def test_plot_signal(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_signal(
            mock_eeg, channel_list=DEAPDataset.channels, sampling_rate=128)

        self.assertFalse(np.all(img == 0))


if __name__ == '__main__':
    unittest.main()
