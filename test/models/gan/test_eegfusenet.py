import unittest

import torch

from torcheeg.models import EEGfuseNet, EFDiscriminator


class TestEEGfuseNet(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):

        g_model = EEGfuseNet(num_electrodes=20, hid_channels_gru=16,
                             num_layers_gru=1, hid_channels_cnn=1, chunk_size=128)
        d_model = EFDiscriminator(
            num_electrodes=20, hid_channels_cnn=1, chunk_size=128)
        X = torch.rand(2, 1, 20, 128)
        fake_X, deep_feature = g_model(X)
        p_real, p_fake = d_model(X), d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2, 64))
        self.assertEqual(tuple(p_real.shape), (2, 1))
        self.assertEqual(tuple(p_fake.shape), (2, 1))

        # other shapes 1
        g_model = EEGfuseNet(num_electrodes=64, hid_channels_gru=16,
                             num_layers_gru=1, hid_channels_cnn=1, chunk_size=512)
        d_model = EFDiscriminator(
            num_electrodes=64, hid_channels_cnn=1, chunk_size=512)
        X = torch.rand(2, 1, 64, 512)
        fake_X, deep_feature = g_model(X)
        p_real, p_fake = d_model(X), d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2, 256))
        self.assertEqual(tuple(p_real.shape), (2, 1))
        self.assertEqual(tuple(p_fake.shape), (2, 1))

        # other shapes 2
        g_model = EEGfuseNet(num_electrodes=32, hid_channels_gru=16,
                             num_layers_gru=1, hid_channels_cnn=1, chunk_size=384)
        d_model = EFDiscriminator(
            num_electrodes=32, hid_channels_cnn=1, chunk_size=384)
        X = torch.randn(2, 1, 32, 384)
        fake_X, deep_feature = g_model(X)
        p_real, p_fake = d_model(X), d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2, 192))
        self.assertEqual(tuple(p_real.shape), (2, 1))
        self.assertEqual(tuple(p_fake.shape), (2, 1))


if __name__ == '__main__':
    unittest.main()
