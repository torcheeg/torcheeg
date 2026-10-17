import copy
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheeg.models import EEGNet
from torcheeg.models.cnn.eegnet import Conv2dWithConstraint


class LegacyConstraint(nn.Conv2d):
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                       maxnorm=1)
        return super().forward(x)


class TestEEGNetConstraints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_projection_values_and_existing_weight_views(self):
        layer = Conv2dWithConstraint(4, 3, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(4)
        view = layer.weight.detach()
        expected = torch.renorm(view.clone(), p=2, dim=0, maxnorm=1)
        x = torch.randn(2, 4, 3, 5)
        output = layer(x)
        torch.testing.assert_close(output, F.conv2d(x, expected))
        torch.testing.assert_close(view, expected)
        self.assertTrue((layer.weight.flatten(1).norm(dim=1) <= 1).all())

    def test_optimizer_steps_match_legacy_projection(self):
        layer = Conv2dWithConstraint(4, 3, 1, bias=False)
        with torch.no_grad():
            layer.weight.mul_(10)
        reference = LegacyConstraint(4, 3, 1, bias=False)
        reference.load_state_dict(layer.state_dict(), strict=True)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9)
        legacy_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1,
                                            momentum=0.9)
        for _ in range(3):
            x = torch.randn(2, 4, 3, 5)
            optimizer.zero_grad()
            legacy_optimizer.zero_grad()
            actual = layer(x)
            expected = reference(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(layer.weight.grad, reference.weight.grad,
                                       rtol=0, atol=0)
            optimizer.step()
            legacy_optimizer.step()
            torch.testing.assert_close(layer.weight, reference.weight,
                                       rtol=0, atol=0)

    def test_multiple_forwards_before_backward(self):
        layer = Conv2dWithConstraint(4, 3, 1, bias=False)
        with torch.no_grad():
            layer.weight.mul_(10)
        reference = LegacyConstraint(4, 3, 1, bias=False)
        reference.load_state_dict(layer.state_dict(), strict=True)
        x = torch.randn(2, 4, 3, 5, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        loss = layer(x).square().mean() + layer(x + 1).square().mean()
        expected = reference(other).square().mean() + reference(other + 1).square().mean()
        loss.backward()
        expected.backward()
        torch.testing.assert_close(x.grad, other.grad)
        torch.testing.assert_close(layer.weight.grad, reference.weight.grad)

    def model(self):
        return EEGNet(num_electrodes=4, chunk_size=128, num_classes=2,
                      dropout=0, F1=2, D=1, F2=2, kernel_1=8, kernel_2=4)

    @unittest.skipUnless(hasattr(torch, 'export') and hasattr(torch.export, 'export'),
                         'torch.export is unavailable in this runtime.')
    def test_strict_inference_export_save_and_load(self):
        training_model = self.model()
        model = copy.deepcopy(training_model).eval().requires_grad_(False)
        x = torch.randn(2, 1, 4, 128)
        # Inference export uses no-grad, including on older exporters.
        with torch.no_grad():
            expected = model(x)
            exported = torch.export.export(model, (x,), strict=True)
            torch.testing.assert_close(exported.module()(x), expected)
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'eegnet.pt2'
                torch.export.save(exported, path)
                restored = torch.export.load(path)
                torch.testing.assert_close(restored.module()(x), expected)
            # Export must not replace live model weights with tracing tensors.
            torch.testing.assert_close(model(x), expected)
        self.assertTrue(all(parameter.requires_grad for parameter in training_model.parameters()))

    def test_state_dict_and_optimizer_parameter_identity(self):
        model = self.model()
        state = copy.deepcopy(model.state_dict())
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        model(torch.randn(2, 1, 4, 128)).square().mean().backward()
        self.assertEqual(identities,
                         {name: id(parameter) for name, parameter in model.named_parameters()})
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = self.model()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(set(state), set(restored.state_dict()))
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
