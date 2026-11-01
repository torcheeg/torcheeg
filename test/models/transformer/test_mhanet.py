import unittest

import torch

from torcheeg.models import MHANet


class TestMhanet(unittest.TestCase):

    def test_mhanet(self):
        eeg = torch.randn(2, 64, 64)
        model = MHANet(num_classes=2, num_electrodes=64, chunk_size=64)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        # other shape
        eeg = torch.randn(2, 32, 128)
        model = MHANet(num_classes=4, num_electrodes=32, chunk_size=128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))

        # other shape
        eeg = torch.randn(2, 32, 1000)
        model = MHANet(num_classes=4, num_electrodes=32, chunk_size=1000)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
