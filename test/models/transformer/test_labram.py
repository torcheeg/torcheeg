import unittest

import torch

from torcheeg.models import LaBraM


class TestLaBraM(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(2, 6, 8, 200)
        model = LaBraM.base_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        model = LaBraM.large_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        model = LaBraM.huge_patch200_200(num_classes=4)
        pred = model(eeg, electrodes=['FP1', 'FPZ', 'FP2',
                                      'AF9', 'AF7', 'AF5'])
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
