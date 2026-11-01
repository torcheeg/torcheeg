import unittest

import torch

from torcheeg.models import LMDA


class TestLmda(unittest.TestCase):

    def test_lmda(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = LMDA(num_electrodes=22,
                     chunk_size=1750,
                     num_classes=4,
                     depth=9,
                     kernel=75)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
