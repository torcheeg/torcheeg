import unittest

import torch

from torcheeg.models import BCDecoder, BCEncoder


class TestBcvae(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        encoder = BCEncoder(in_channels=4, num_classes=3)
        decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=3)
        y = torch.randint(low=0, high=3, size=(1, ))
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z, y)
        for output in fake_X if isinstance(fake_X, (tuple, list)) else (fake_X,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())

        self.assertEqual(tuple(fake_X.shape), (1, 4, 9, 9))


if __name__ == '__main__':
    unittest.main()
