import unittest

import torch

from torcheeg.models import LaBraM


class TestLabram(unittest.TestCase):
    def test_labram(self):
        eeg = torch.randn(2, 6, 8, 200)
        model = LaBraM.base_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        model = LaBraM.large_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        model = LaBraM.huge_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
