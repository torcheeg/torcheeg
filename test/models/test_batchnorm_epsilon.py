import io
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheeg.models import ATCNet, EEGfuseNet, EFDiscriminator


class TestBatchNormEpsilon(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def models(self):
        return [EEGfuseNet(num_electrodes=20, chunk_size=128),
                EFDiscriminator(num_electrodes=20, chunk_size=128),
                ATCNet(num_electrodes=32, chunk_size=256, num_classes=3)]

    def test_batch_norm_stabilizes_constant_features_and_keeps_affine_parameters(self):
        for model in self.models():
            for name, layer in model.named_modules():
                if isinstance(layer, nn.BatchNorm2d):
                    with self.subTest(model=type(model).__name__, layer=name):
                        self.assertEqual(layer.eps, 1e-5)
                        self.assertTrue(layer.affine)
                        x = torch.ones(2, layer.num_features, 2, 3,
                                       requires_grad=True)
                        output = layer(x)
                        self.assertTrue(torch.isfinite(output).all())
                        output.square().mean().backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        self.assertTrue(torch.isfinite(layer.running_var).all())

    def test_atcnet_training_and_constant_input(self):
        model = ATCNet(num_electrodes=32, chunk_size=256, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        weight = model.conv_block[0].weight.detach().clone()
        for x in [torch.randn(2, 1, 32, 256), torch.zeros(2, 1, 32, 256)]:
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, torch.tensor([0, 2]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertFalse(torch.equal(weight, model.conv_block[0].weight))
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(torch.zeros(2, 1, 32, 256))).all())

    def test_eegfusenet_adversarial_and_reconstruction_gradients(self):
        generator = EEGfuseNet(num_electrodes=20, chunk_size=128)
        discriminator = EFDiscriminator(num_electrodes=20, chunk_size=128)
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        before = generator.conv1.weight.detach().clone()
        for x in [torch.randn(2, 1, 20, 128), torch.zeros(2, 1, 20, 128)]:
            optimizer.zero_grad()
            discriminator.zero_grad()
            reconstructed, features = generator(x)
            score = discriminator(reconstructed)
            loss = F.mse_loss(reconstructed, x) + features.square().mean()
            loss = loss + F.binary_cross_entropy(score, torch.ones_like(score))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for model in [generator, discriminator]:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertIsNotNone(generator.gru_en.weight_ih_l0.grad)
            self.assertIsNotNone(discriminator.conv1.weight.grad)
            optimizer.step()
        self.assertFalse(torch.equal(before, generator.conv1.weight))
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            output, _ = generator(torch.zeros(2, 1, 20, 128))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(discriminator(output)).all())

    def test_state_dict_round_trip_preserves_parameters_and_buffers(self):
        for model, restored in zip(self.models(), self.models()):
            with self.subTest(model=type(model).__name__):
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                state = torch.load(checkpoint, weights_only=True)
                restored.load_state_dict(state, strict=True)
                self.assertEqual(set(state), set(restored.state_dict()))
                for name, value in restored.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
