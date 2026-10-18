from argparse import Namespace
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import SleepDG
from test.models.fixtures.sleepdg_reference.models.model import Model as AuthorModel


def small_model(**kwargs):
    config = dict(sequence_length=2, hid_channels=8, num_heads=2, mlp_channels=16, dropout=0.0)
    config.update(kwargs)
    return SleepDG(**config)


def objective(outputs):
    return sum(
        (
            x * torch.linspace(-0.7, 1.1, x.numel(), dtype=x.dtype, device=x.device).reshape_as(x)
        ).mean()
        for x in outputs
    )


class TestSleepDG(unittest.TestCase):
    def test_independent_author_full_model_outputs_and_every_gradient(self):
        torch.manual_seed(42)
        reference = AuthorModel(Namespace(dropout=0.0, num_of_classes=5)).double().eval()
        torch.manual_seed(42)
        model = SleepDG(dropout=0.0).double().eval()
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        for n, t in model.state_dict().items():
            torch.testing.assert_close(t, reference.state_dict()[n], rtol=0, atol=0)
        x = torch.randn(1, 20, 2, 3000, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        actual, expected = model.forward_all(x), reference(other)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
        objective(actual).backward()
        objective(expected).backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-8, atol=1e-10)
        for (n, p), (m, q) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(n, m)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10, msg=n)
        with torch.no_grad():
            torch.testing.assert_close(
                model(x.detach()), reference.inference(x.detach()), rtol=1e-10, atol=1e-11
            )

    def test_author_training_dropout_and_all_batchnorm_buffers_match(self):
        torch.manual_seed(42)
        reference = AuthorModel(Namespace(dropout=0.1, num_of_classes=5)).double().train()
        torch.manual_seed(42)
        model = SleepDG().double().train()
        x = torch.randn(1, 20, 2, 3000, dtype=torch.float64)
        torch.manual_seed(123)
        expected = reference(x)
        torch.manual_seed(123)
        actual = model.forward_all(x)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
        for n, t in model.state_dict().items():
            torch.testing.assert_close(t, reference.state_dict()[n], rtol=0, atol=0)

    def test_author_reference_provenance_and_package_local_imports(self):
        root = Path(__file__).parent / 'fixtures' / 'sleepdg_reference'
        manifest = json.loads((root / 'source-manifest.json').read_text())
        self.assertEqual(manifest['revision'], '01d209d4fda874933a277a88bccdca6e3835e884')
        self.assertEqual(len(manifest['files']), 6)
        for item in manifest['files']:
            self.assertEqual(
                hashlib.sha256((root / item['path']).read_bytes()).hexdigest(),
                item['adapted_sha256'],
            )
        self.assertIn('fixtures.sleepdg_reference.models', AuthorModel.__module__)

    def test_single_batch_epoch_hidden_and_configured_channel_axes_are_preserved(self):
        for channels, width, heads in ((1, 1, 1), (2, 8, 2), (3, 12, 3)):
            model = small_model(
                num_electrodes=channels, sequence_length=1, hid_channels=width, num_heads=heads
            ).eval()
            x = torch.randn(1, 1, channels, 3000)
            logits, reconstruction, features = model.forward_all(x)
            self.assertEqual(tuple(logits.shape), (1, 1, 5))
            self.assertEqual(tuple(reconstruction.shape), (1, 1, channels, 3000))
            self.assertEqual(tuple(features.shape), (1, 1, width))
            self.assertEqual(model.feature_dim(), width)
            for output in (logits, reconstruction, features):
                self.assertTrue(torch.isfinite(output).all())

    def test_inference_omits_decoder_and_constructor_does_not_update_batchnorm(self):
        model = small_model().train()
        counters = {
            n: t.clone() for n, t in model.named_buffers() if n.endswith('num_batches_tracked')
        }
        self.assertEqual(len(counters), 9)
        self.assertTrue(all(t.item() == 0 for t in counters.values()))
        x = torch.randn(2, 2, 2, 3000)
        with patch.object(
            model.ae.decoder, 'forward', side_effect=AssertionError('decoder must not run')
        ):
            model(x)
            model.feature(x)
        for n, t in model.named_buffers():
            if n.endswith('num_batches_tracked'):
                self.assertEqual(t.item(), 0 if 'decoder' in n else 2)
        model.forward_all(x)
        for n, t in model.named_buffers():
            if n.endswith('num_batches_tracked'):
                self.assertEqual(t.item(), 1 if 'decoder' in n else 3)

    def test_sequence_context_is_across_epochs_and_eval_does_not_mix_samples(self):
        model = small_model(sequence_length=3).eval()
        x = torch.randn(2, 3, 2, 3000)
        expected = model(x)
        changed = x.clone()
        changed[0, 0] *= 3
        actual = model(changed)
        self.assertFalse(torch.allclose(actual[0, 1:], expected[0, 1:], rtol=1e-6, atol=1e-8))
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_full_optimizer_updates_encoder_decoder_projection_and_classifier(self):
        model = small_model().train()
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(2, 2, 2, 3000, requires_grad=True)
        labels = torch.tensor([[0, 1], [2, 4]])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits, reconstruction, features = model.forward_all(x)
            loss = (
                F.cross_entropy(logits.flatten(0, 1), labels.flatten())
                + F.mse_loss(reconstruction, x)
                + features.square().mean()
            )
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().max().item(), 0.0)
            for n, p in model.named_parameters():
                self.assertIsNotNone(p.grad, n)
                self.assertTrue(torch.isfinite(p.grad).all(), n)
            optimizer.step()
        for n in (
            'ae.encoder.epoch_encoder.encoder.0.weight',
            'ae.encoder.seq_encoder.pos_embedding',
            'ae.encoder.fc_mu.weight',
            'ae.decoder.upsample.0.weight',
            'ae.decoder.upsample.20.weight',
            'classifier.weight',
        ):
            self.assertFalse(torch.equal(dict(model.named_parameters())[n], initial[n]), n)

    def test_complete_tensor_checkpoint_strictly_restores_buffers_and_outputs(self):
        model = small_model().train()
        x = torch.randn(2, 2, 2, 3000)
        model.forward_all(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = small_model().eval()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        model.eval()
        for left, right in zip(model.forward_all(x), restored.forward_all(x)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for n, t in model.state_dict().items():
            torch.testing.assert_close(t, restored.state_dict()[n], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            small_model(sequence_length=1).load_state_dict(model.state_dict(), strict=True)

    def test_noncontiguous_input_and_device_dtype_moves(self):
        model = small_model().double().eval()
        x = torch.randn(2, 2, 2, 6000, dtype=torch.float64)[..., ::2]
        self.assertFalse(x.is_contiguous())
        for left, right in zip(model.forward_all(x), model.forward_all(x.contiguous())):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(
            all(
                t.dtype == torch.float64
                for t in model.state_dict().values()
                if t.is_floating_point()
            )
        )
        model.float()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            outputs = model.forward_all(x.float())
        self.assertTrue(all(torch.isfinite(t).all() for t in outputs))

    def test_invalid_shapes_types_and_configuration_are_explicit(self):
        for name in (
            'num_electrodes',
            'chunk_size',
            'sequence_length',
            'num_classes',
            'hid_channels',
            'num_layers',
            'num_heads',
            'mlp_channels',
        ):
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    small_model(**{name: value})
        for kwargs in (
            dict(chunk_size=2999),
            dict(hid_channels=9),
            dict(dropout=1.0),
            dict(attention_dropout=float('nan')),
            dict(dropout=True),
        ):
            with self.assertRaises(ValueError):
                small_model(**kwargs)
        model = small_model()
        for x in (
            None,
            torch.randn(2, 2, 3000),
            torch.randn(0, 2, 2, 3000),
            torch.randn(1, 1, 2, 3000),
            torch.randn(1, 2, 3, 3000),
            torch.randn(1, 2, 2, 2999),
            torch.zeros(1, 2, 2, 3000, dtype=torch.int64),
            torch.zeros(1, 2, 2, 3000, dtype=torch.complex64),
        ):
            with self.assertRaises(ValueError):
                model(x)
            with self.assertRaises(ValueError):
                model.forward_all(x)

    @unittest.skipUnless(
        os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('TORCHEEG_TEST_COMPILE') == '1',
        'Optional full-graph compiler regression',
    )
    def test_fullgraph_classification_and_reconstruction_training_match_all_gradients(self):
        for full in (False, True):
            model = small_model(sequence_length=3).eval()
            reference = copy.deepcopy(model)
            x = torch.randn(2, 3, 2, 3000, requires_grad=True)
            other = x.detach().clone().requires_grad_()

            def forward(value):
                return model.forward_all(value) if full else model(value)

            actual = torch.compile(forward, fullgraph=True, dynamic=False)(x)
            expected = reference.forward_all(other) if full else reference(other)
            if not full:
                actual, expected = (actual,), (expected,)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=5e-4, atol=3e-5)
            objective(actual).backward()
            objective(expected).backward()
            torch.testing.assert_close(x.grad, other.grad, rtol=3e-3, atol=1e-5)
            for (n, p), (m, q) in zip(model.named_parameters(), reference.named_parameters()):
                self.assertEqual(n, m)
                if q.grad is None:
                    self.assertIsNone(p.grad, n)
                else:
                    torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=1e-5, msg=n)


if __name__ == '__main__':
    unittest.main()
