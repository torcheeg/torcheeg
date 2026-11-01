import unittest

import torch

from torcheeg.models import CSPNet


class TestCSPNet(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = CSPNet(chunk_size=1750,
                       num_electrodes=22,
                       num_classes=5,
                       num_filters_t=20,
                       filter_size_t=25)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
