import unittest

import torch

from torcheeg.models import USleep


class TestUSleep(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(2, 1, 3000)
        model = USleep(num_electrodes=1,
                       patch_size=100,
                       num_patchs=30,
                       num_classes=5)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
