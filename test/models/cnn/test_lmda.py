import unittest

import torch

from torcheeg.models import LMDA


class TestLMDA(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = LMDA(num_electrodes=22,
                     chunk_size=1750,
                     num_classes=4,
                     depth=9,
                     kernel=75)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
