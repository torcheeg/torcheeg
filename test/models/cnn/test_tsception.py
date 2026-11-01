import unittest

import torch

from torcheeg.models import TSCeption


class TestTsception(unittest.TestCase):

    def test_tsception(self):
        eeg = torch.randn(1, 1, 28, 512)
        model = TSCeption(num_classes=2,
                          num_electrodes=28,
                          sampling_rate=128,
                          num_T=15,
                          num_S=15,
                          hid_channels=32,
                          dropout=0.5)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
