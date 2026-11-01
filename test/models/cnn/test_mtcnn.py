import unittest

import torch

from torcheeg.models import MTCNN


class TestMtcnn(unittest.TestCase):

    def test_mtcnn(self):
        eeg = torch.randn(1, 8, 8, 9)
        model = MTCNN(num_classes=2,
                      in_channels=8,
                      grid_size=(8, 9),
                      dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred[0].shape), (1, 2))
        self.assertEqual(tuple(pred[1].shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
