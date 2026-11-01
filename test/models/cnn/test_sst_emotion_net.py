import unittest

import torch

from torcheeg.models import SSTEmotionNet


class TestSstEmotionNet(unittest.TestCase):

    def test_sst_emotion_net(self):
        eeg = torch.randn(2, 32 + 4, 16, 16)
        model = SSTEmotionNet(temporal_in_channels=32,
                              spectral_in_channels=4,
                              grid_size=(16, 16),
                              num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
