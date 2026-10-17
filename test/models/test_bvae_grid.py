import io
import unittest

import torch
from torch import nn

from torcheeg.models import BDecoder, BEncoder


class TestBVAEGrid(unittest.TestCase):

    def test_encoder_initialization_does_not_train_batch_norm(self):
        encoder = BEncoder(hid_channels=8)
        for module in encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                torch.testing.assert_close(module.running_mean, torch.zeros_like(module.running_mean))
                torch.testing.assert_close(module.running_var, torch.ones_like(module.running_var))
                self.assertEqual(module.num_batches_tracked.item(), 0)

    def test_feature_dimension_query_preserves_state_and_mode(self):
        encoder = BEncoder(hid_channels=8, grid_size=(8, 10))
        encoder.train()
        before = {key: value.clone() for key, value in encoder.state_dict().items()}
        dimension = encoder.feature_dim
        self.assertEqual(encoder.feature_dim, dimension)
        self.assertTrue(encoder.training)
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[key])
        encoder.eval()
        with torch.no_grad():
            mean, _ = encoder(torch.zeros(2, 4, 8, 10))
        self.assertEqual(mean.shape, (2, 8))

    def test_requested_grid_shapes_reconstruct_and_backpropagate(self):
        for shape in ((1, 1), (4, 4), (5, 5), (8, 10), (9, 9), (16, 17)):
            with self.subTest(shape=shape):
                encoder = BEncoder(in_channels=3, hid_channels=8, grid_size=shape)
                decoder = BDecoder(in_channels=8, out_channels=3, grid_size=shape)
                eeg = torch.randn(2, 3, *shape)
                mean, logvar = encoder(eeg)
                latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                reconstruction = decoder(latent)
                self.assertEqual(reconstruction.shape, eeg.shape)
                loss = (reconstruction - eeg).square().mean()
                loss += 0.01 * (mean.square() + logvar.exp() - logvar - 1).mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for model in (encoder, decoder):
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_default_and_rectangular_checkpoint_round_trips(self):
        for shape in ((9, 9), (5, 8)):
            with self.subTest(shape=shape):
                encoder = BEncoder(hid_channels=8, grid_size=shape).eval()
                decoder = BDecoder(in_channels=8, grid_size=shape).eval()
                eeg = torch.randn(2, 4, *shape)
                with torch.no_grad():
                    mean, logvar = encoder(eeg)
                    expected = decoder(mean)
                buffer = io.BytesIO()
                torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, buffer)
                buffer.seek(0)
                state = torch.load(buffer, weights_only=True)
                restored_encoder = BEncoder(hid_channels=8, grid_size=shape).eval()
                restored_decoder = BDecoder(in_channels=8, grid_size=shape).eval()
                restored_encoder.load_state_dict(state['encoder'], strict=True)
                restored_decoder.load_state_dict(state['decoder'], strict=True)
                with torch.no_grad():
                    actual_mean, actual_logvar = restored_encoder(eeg)
                    actual = restored_decoder(actual_mean)
                torch.testing.assert_close(actual_mean, mean)
                torch.testing.assert_close(actual_logvar, logvar)
                torch.testing.assert_close(actual, expected)

    def test_decoder_output_depends_on_latent_values(self):
        decoder = BDecoder(in_channels=8, grid_size=(5, 5)).eval()
        latent = torch.randn(2, 8, requires_grad=True)
        output = decoder(latent)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0)

    def test_invalid_grid_sizes_have_clear_errors(self):
        for shape in ((0, 9), (9, -1), (9,), (9, 9, 9), (9., 9), (True, 9), None):
            for model in (BEncoder, BDecoder):
                with self.subTest(shape=shape, model=model.__name__):
                    with self.assertRaisesRegex(ValueError, 'grid_size'):
                        model(grid_size=shape)


if __name__ == '__main__':
    unittest.main()
