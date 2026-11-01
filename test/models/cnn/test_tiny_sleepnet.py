import unittest

import torch

from torcheeg.models import TinySleepNet


class TestTinySleepnet(unittest.TestCase):

    def test_tiny_sleepnet(self):
        eeg = torch.randn(2, 1, 3000, 1)
        model = TinySleepNet(num_classes=5,
                             chunk_size=3000,
                             num_electrodes=1)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
