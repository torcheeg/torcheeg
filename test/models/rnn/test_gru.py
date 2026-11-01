import unittest

import torch

from torcheeg.models import GRU


class TestGru(unittest.TestCase):
    def test_gru(self):
        model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

        eeg = torch.randn(2, 32, 128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
