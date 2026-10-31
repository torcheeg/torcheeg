"""Functional BiDANN boundaries, input derivatives and adversarial loss."""

import unittest
import torch
from torch.nn import functional as F
from torcheeg.models import BiDANN, BiDANNLoss
from torcheeg.models.rnn.bidann import (
    _GradientReverse,
    _split_hemispheres,
    _SEED_CHANNELS,
    _LEFT_CHANNELS,
    _RIGHT_CHANNELS,
)


class TestBiDANN(unittest.TestCase):
    def test_parameter_input_and_loss_validation(self):
        for config in [
            dict(num_electrodes=32),
            dict(in_channels=0),
            dict(num_classes=True),
            dict(sequence_length=0),
            dict(adversarial_weight=-1),
            dict(adversarial_weight=float('nan')),
        ]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    BiDANN(**config)
        for weight in [-1, float('inf'), True]:
            with self.assertRaises(ValueError):
                BiDANNLoss(weight)
        model = BiDANN()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 62, 5))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 9, 61, 5))
        with self.assertRaises(TypeError):
            model(torch.ones(2, 9, 62, 5, dtype=torch.int64))
        with self.assertRaises(ValueError):
            BiDANNLoss()((torch.randn(2, 3),), torch.tensor([0, 1]))

    def test_requested_class_count_and_complete_head(self):
        for classes in [2, 4]:
            model = BiDANN(num_classes=classes).eval()
            source = torch.randn(3, 9, 62, 5)
            target = torch.randn(2, 9, 62, 5)
            outputs = model(source, target)
            self.assertEqual(outputs[0].shape, (3, classes))
            self.assertEqual([tuple(x.shape) for x in outputs[1:]], [(5, 2)] * 3)
            BiDANNLoss()(outputs, torch.tensor([0, 1, classes - 1])).backward()
            for name, value in model.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertIs(model.classifier, model.classifer)
            self.assertFalse(
                any(name.startswith('classifier.') for name in model.state_dict())
            )
            with self.assertRaises(RuntimeError):
                model.load_state_dict({}, strict=True)

    def test_gradient_reversal_identity_and_exact_backward(self):
        for coefficient in [0.0, 0.3, 1.0]:
            x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
            gradient = torch.randn_like(x)
            y = _GradientReverse.apply(x, coefficient)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            y.backward(gradient)
            torch.testing.assert_close(x.grad, -coefficient * gradient, rtol=0, atol=0)

    def test_independent_input_finite_differences_without_domain_path(self):
        torch.manual_seed(23)
        model = BiDANN(in_channels=2, sequence_length=3).double().eval()
        x = torch.randn(2, 3, 62, 2, dtype=torch.float64, requires_grad=True)
        weights = torch.tensor(
            [[0.2, 0.3, -0.4], [0.5, -0.1, 0.7]], dtype=torch.float64
        )
        (model(x) * weights).sum().backward()
        for index in [(0, 0, 0, 0), (0, 1, 2, 1), (1, 2, 1, 1), (1, 0, 9, 0)]:
            plus = x.detach().clone()
            minus = x.detach().clone()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            with torch.no_grad():
                expected = (
                    (model(plus) * weights).sum() - (model(minus) * weights).sum()
                ) / 2e-5
            torch.testing.assert_close(x.grad[index], expected, rtol=1e-5, atol=1e-7)

    def test_domain_labels_and_native_four_cross_entropy_sum(self):
        scores = [torch.randn(3, 4, requires_grad=True)] + [
            torch.randn(5, 2, requires_grad=True) for _ in range(3)
        ]
        labels = torch.tensor([0, 1, 3])
        domain = torch.tensor([0, 0, 0, 1, 1])
        expected = F.cross_entropy(scores[0], labels)
        for head in scores[1:]:
            expected = expected + F.cross_entropy(head, domain)
        actual = BiDANNLoss()(scores, labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            BiDANNLoss(0)(scores, labels),
            F.cross_entropy(scores[0], labels),
            rtol=0,
            atol=0,
        )

    def test_aligned_frames_keep_order_and_share_training_inference_layout(self):
        x = (
            torch.arange(2 * 3 * 62 * 2, dtype=torch.float32)
            .reshape(2, 3, 62, 2)
            .requires_grad_()
        )
        left, right = _split_hemispheres(
            x, _SEED_CHANNELS, _LEFT_CHANNELS, _RIGHT_CHANNELS, layout='aligned'
        )
        expected_left = x[
            :, :, [_SEED_CHANNELS.index(n) for n in _LEFT_CHANNELS], :
        ].reshape(2, 3, -1)
        expected_right = x[
            :, :, [_SEED_CHANNELS.index(n) for n in _RIGHT_CHANNELS], :
        ].reshape(2, 3, -1)
        torch.testing.assert_close(left, expected_left, rtol=0, atol=0)
        torch.testing.assert_close(right, expected_right, rtol=0, atol=0)
        legacy = _split_hemispheres(
            x, _SEED_CHANNELS, _LEFT_CHANNELS, _RIGHT_CHANNELS, layout='libeer'
        )[1]
        self.assertFalse(torch.equal(legacy, right))
        (left.sum() + right.sum()).backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
        model = BiDANN().eval()
        source = torch.randn(3, 9, 62, 5)
        target = torch.randn(2, 9, 62, 5)
        with torch.no_grad():
            torch.testing.assert_close(
                model(source, target)[0], model(source), rtol=0, atol=0
            )
        with self.assertRaises(ValueError):
            BiDANN(hemisphere_layout='unknown')
