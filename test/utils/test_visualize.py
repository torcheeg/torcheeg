import unittest

import torch
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LIST
from torcheeg.utils.visualize import (plot_feature_topomap, plot_raw_topomap,
                                      plot_signal, plot_3d_tensor,
                                      plot_2d_tensor)


class TestVisualize(unittest.TestCase):
    def test_plot_raw_topomap(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_raw_topomap(mock_eeg,
                               channel_list=DEAP_CHANNEL_LIST,
                               sampling_rate=128)
        self.assertEqual(img.shape, (347, 1550, 4))

    def test_plot_feature_topomap(self):
        mock_eeg = torch.randn(32, 4)
        img = plot_feature_topomap(
            mock_eeg,
            channel_list=DEAP_CHANNEL_LIST,
            feature_list=["theta", "alpha", "beta", "gamma"])
        self.assertEqual(img.shape, (347, 1550, 4))

    def test_plot_signal(self):
        mock_eeg = torch.randn(32, 128)
        img = plot_signal(mock_eeg,
                          channel_list=DEAP_CHANNEL_LIST,
                          sampling_rate=128)
        self.assertEqual(img.shape, (773, 727, 4))

    def test_plot_3d_tensor(self):
        mock_eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(mock_eeg)
        self.assertEqual(img.shape, (374, 382, 4))

    def test_plot_2d_tensor(self):
        mock_eeg = torch.randn(9, 9)
        img = plot_2d_tensor(mock_eeg)
        self.assertEqual(img.shape, (393, 388, 4))


if __name__ == '__main__':
    unittest.main()
