import unittest

import torch

from torcheeg.models import TCNet


class TestTcnet(unittest.TestCase):

    def test_tcnet(self):
        eeg = torch.randn(2, 22, 1000)
        model = TCNet(num_classes=4,
                      num_electrodes=22,
                      F1=8,
                      D=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
