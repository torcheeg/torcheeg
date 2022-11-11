import unittest

import torch
from torcheeg.models import BCEncoder, BCDecoder, BEncoder, BDecoder


class TestVAE(unittest.TestCase):
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

        encoder = encoder.cuda()
        decoder = decoder.cuda()
        mock_eeg = mock_eeg.cuda()
        mu, logvar = encoder(mock_eeg)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z)

        self.assertEqual(tuple(fake_X.shape), (1, 4, 9, 9))

    def test_bcvae(self):
        encoder = BCEncoder(in_channels=4, num_classes=3)
        decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=3)
        y = torch.randint(low=0, high=3, size=(1, ))
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z, y)

        self.assertEqual(tuple(fake_X.shape), (1, 4, 9, 9))

        encoder = encoder.cuda()
        decoder = decoder.cuda()
        y = y.cuda()
        mock_eeg = mock_eeg.cuda()

        mu, logvar = encoder(mock_eeg, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z, y)
        self.assertEqual(tuple(fake_X.shape), (1, 4, 9, 9))


if __name__ == '__main__':
    unittest.main()
