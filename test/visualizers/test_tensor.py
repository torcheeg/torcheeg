import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from torcheeg.visualizers import plot_2d_tensor, plot_3d_tensor


class TestTensor(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

    def test_plot_3d_tensor(self):
        mock_eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(mock_eeg)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(np.ptp(img[..., :3]), 0)

    def test_plot_2d_tensor(self):
        mock_eeg = torch.randn(9, 9)
        img = plot_2d_tensor(mock_eeg)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(np.ptp(img[..., :3]), 0)


if __name__ == '__main__':
    unittest.main()
