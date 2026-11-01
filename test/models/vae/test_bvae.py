import unittest

import torch

from torcheeg.models import BDecoder, BEncoder


class TestBvae(unittest.TestCase):
    def test_bvae(self):
        encoder = BEncoder(in_channels=4)
        decoder = BDecoder(in_channels=64, out_channels=4)
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z)

        self.assertEqual(tuple(fake_X.shape), (1, 4, 9, 9))


if __name__ == '__main__':
    unittest.main()
