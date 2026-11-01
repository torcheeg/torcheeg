import unittest

import torch

from torcheeg.models import DGCNN


class TestDgcnn(unittest.TestCase):

    def test_dgcnn(self):
        eeg = torch.randn(1, 62, 200)
        model = DGCNN(in_channels=200, num_electrodes=62,
                      hid_channels=32, num_layers=2, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
