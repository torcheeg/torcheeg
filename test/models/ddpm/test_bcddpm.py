import unittest

import torch

from torcheeg.models import BCUNet


class TestBcddpm(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        unet = BCUNet(num_classes=2)
        mock_eeg = torch.randn(2, 4, 9, 9)
        t = torch.randint(low=1, high=1000, size=(2, ))
        y = torch.randint(low=0, high=2, size=(1, ))
        fake_X = unet(mock_eeg, t, y)
        for output in fake_X if isinstance(fake_X, (tuple, list)) else (fake_X,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(fake_X.shape), (2, 4, 9, 9))


if __name__ == '__main__':
    unittest.main()
