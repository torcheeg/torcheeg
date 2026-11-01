import unittest

import torch

from torcheeg.models import ATCNet


class TestAtcnet(unittest.TestCase):

    def test_atcnet(self):
        eeg = torch.randn(2, 1, 32, 1000)
        model = ATCNet(num_classes=3, num_electrodes=32, chunk_size=1000)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 3))

        # other shape
        eeg = torch.randn(2, 1, 64, 512)
        model = ATCNet(num_classes=4, num_electrodes=64, chunk_size=512)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))

        # other shape
        eeg = torch.rand(2, 1, 96, 256)
        model = ATCNet(num_classes=4, num_electrodes=96, chunk_size=256)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
