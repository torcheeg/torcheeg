import unittest

import torch

from torcheeg.models import CCNN


class TestCcnn(unittest.TestCase):

    def test_ccnn(self):
        eeg = torch.randn(1, 4, 9, 9)
        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
