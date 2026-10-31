"""Functional contracts for hierarchical spatial learning."""

import unittest
import torch
from torcheeg.models import HSLT


class TestHSLT(unittest.TestCase):
    def test_parameter_and_input_validation(self):
        for config in [
            dict(num_electrodes=31),
            dict(num_electrodes=0),
            dict(in_channels=True),
            dict(electrode_depth=0),
            dict(region_embed_dim=-1),
            dict(num_classes=0),
        ]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    HSLT(**config)
        with self.assertRaises(TypeError):
            HSLT(return_logits=1)
        model = HSLT()
        for x in [
            torch.randn(2, 31, 5),
            torch.randn(2, 32, 4),
            torch.randn(2, 1, 32, 5),
        ]:
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(TypeError):
            model(torch.ones(2, 32, 5, dtype=torch.int64))

    def test_native_activation_and_opt_in_logits(self):
        for channels, classes in [(32, 2), (62, 3)]:
            model = HSLT(num_electrodes=channels, num_classes=classes).eval()
            logits_model = HSLT(
                num_electrodes=channels, num_classes=classes, return_logits=True
            ).eval()
            logits_model.load_state_dict(model.state_dict(), strict=True)
            x = torch.randn(2, channels, 5)
            with torch.no_grad():
                actual = model(x)
                logits = logits_model(x)
            expected = logits.sigmoid() if classes == 2 else logits.softmax(-1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, (2, classes))
            self.assertTrue(torch.isfinite(actual).all())

    def test_independent_samples_strict_state_and_cross_entropy(self):
        model = HSLT(return_logits=True).eval()
        x = torch.randn(3, 32, 5)
        with torch.no_grad():
            together = model(x)
            alone = model(x[:1])
        torch.testing.assert_close(together[:1], alone, rtol=1e-5, atol=1e-6)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 0]))
        loss.backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        with self.assertRaises(RuntimeError):
            model.load_state_dict({}, strict=True)
