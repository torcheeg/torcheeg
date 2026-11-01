import unittest

import torch

from torcheeg.models import Conformer


class TestConformer(unittest.TestCase):

    def test_conformer(self):
        eeg = torch.randn(2, 1, 32, 128)
        model = Conformer(num_electrodes=32, sampling_rate=128, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
