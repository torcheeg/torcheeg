import unittest

import torch

from torcheeg.models import STNet


class TestSTNet(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = STNet(num_classes=2,
                      chunk_size=128,
                      grid_size=(9, 9),
                      dropout=0.2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
