import unittest

import numpy as np
import torch

from torcheeg.visualizers import plot_2d_tensor, plot_3d_tensor


class TestTensor(unittest.TestCase):

    def test_plot_3d_tensor(self):
        mock_eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(mock_eeg)

        self.assertFalse(np.all(img == 0))

    def test_plot_2d_tensor(self):
        mock_eeg = torch.randn(9, 9)
        img = plot_2d_tensor(mock_eeg)

        self.assertFalse(np.all(img == 0))


if __name__ == '__main__':
    unittest.main()
