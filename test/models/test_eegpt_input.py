import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import resample
import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import EEGPT, EEGPTClassifier, EEGPTInputAdapter, EEGPTResample, EEGPTAdaptedModel


class TestEEGPTResample(unittest.TestCase):
    def test_matches_scipy_across_odd_even_lengths_and_ranks(self):
        rng = np.random.RandomState(42)
        for source in [1, 2, 7, 8, 9, 16]:
            for target in [1, 2, 5, 8, 13, 16]:
                with self.subTest(source=source, target=target):
                    values = rng.normal(size=(2, 3, source))
                    actual = EEGPTResample(target)(torch.tensor(values))
                    np.testing.assert_allclose(actual.numpy(), resample(values, target, axis=-1), rtol=1e-12, atol=1e-12)
        for shape in [(7,), (2, 7), (2, 3, 4, 7)]:
            values = rng.normal(size=shape)
            np.testing.assert_allclose(EEGPTResample(12)(torch.tensor(values)).numpy(),
                                       resample(values, 12, axis=-1), rtol=1e-12, atol=1e-12)

    def test_nyquist_amplitude_and_removed_high_frequencies(self):
        cases = [(8, 16, 4), (8, 4, 2), (7, 8, 3), (9, 6, 3)]
        for source, target, frequency in cases:
            with self.subTest(source=source, target=target):
                values = torch.cos(2 * torch.pi * frequency * torch.arange(source, dtype=torch.float64) / source)
                expected = torch.cos(2 * torch.pi * frequency * torch.arange(target, dtype=torch.float64) / target)
                torch.testing.assert_close(EEGPTResample(target)(values), expected, rtol=1e-12, atol=1e-12)
        high = torch.cos(2 * torch.pi * 3 * torch.arange(8, dtype=torch.float64) / 8)
        torch.testing.assert_close(EEGPTResample(4)(high), torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12)
        torch.testing.assert_close(EEGPTResample(13)(torch.full((2, 3, 8), 2.5)), torch.full((2, 3, 13), 2.5))

    def test_gradient_matches_scipy_linear_operator(self):
        for source, target in [(7, 8), (8, 7), (4, 8), (8, 4), (1, 9), (9, 1)]:
            with self.subTest(source=source, target=target):
                x = torch.randn(2, 3, source, dtype=torch.float64, requires_grad=True)
                other = x.detach().clone().requires_grad_(True)
                matrix = torch.tensor(resample(np.eye(source), target, axis=-1))
                expected = other @ matrix
                actual = EEGPTResample(target)(x)
                coefficient = torch.linspace(.1, 1, actual.numel(), dtype=torch.float64).reshape(actual.shape)
                (actual * coefficient).sum().backward()
                (expected * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, other.grad, rtol=1e-10, atol=1e-12)
                value = torch.randn(2, source, dtype=torch.float64, requires_grad=True)
                EEGPTResample(target)(value).sum().backward()
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_gradcheck_and_noncontiguous_input(self):
        for source, target in [(5, 8), (8, 5), (4, 8), (8, 4)]:
            with self.subTest(source=source, target=target):
                x = torch.randn(1, 2, source, dtype=torch.float64, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(EEGPTResample(target), (x,), atol=1e-6, rtol=1e-5))
        x = torch.randn(2, 3, 14, dtype=torch.float64)[..., ::2]
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(EEGPTResample(12)(x), torch.tensor(resample(x.numpy(), 12, axis=-1)), rtol=1e-12, atol=1e-12)

    def test_identity_storage_and_low_precision_cpu_gradients(self):
        x = torch.randn(2, 3, 8)
        self.assertIs(EEGPTResample(8)(x), x)
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 3, 7).to(dtype).requires_grad_(True)
                result = EEGPTResample(12)(x)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, x.device)
                expected = torch.tensor(resample(x.detach().float().numpy(), 12, axis=-1)).to(dtype)
                torch.testing.assert_close(result, expected, rtol=.01, atol=.01)
                result.float().square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_samples_and_input_types(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                EEGPTResample(value)
        for x in [None, [1, 2], torch.tensor(1.), torch.empty(0), torch.empty(0, 3, 8),
                  torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.complex64)]:
            with self.subTest(type=type(x)), self.assertRaises(ValueError):
                EEGPTResample(8)(x)


class TestEEGPTInputAdapter(unittest.TestCase):
    def encoder(self):
        return EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'], chunk_size=128,
                     patch_size=32, hid_channels=16, embed_num=2, depth=2, num_heads=4)

    def test_shapes_and_clean_constructor_batchnorm(self):
        adapter = EEGPTInputAdapter(19, 3, num_samples=128)
        norms = [module for module in adapter.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertEqual(len(norms), 2)
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.item(), 0)
            torch.testing.assert_close(norm.running_mean, torch.zeros_like(norm.running_mean))
            torch.testing.assert_close(norm.running_var, torch.ones_like(norm.running_var))
        adapter.eval()
        for length in [101, 128, 200]:
            with self.subTest(length=length):
                self.assertEqual(adapter(torch.randn(2, 19, length)).shape, (2, 3, 128))
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.item(), 0)
        adapter.train()
        adapter(torch.randn(2, 19, 101))
        self.assertTrue(all(norm.num_batches_tracked.item() == 1 for norm in norms))

    def test_full_pipeline_matches_independent_resampling_convolutions_and_gradients(self):
        adapter = EEGPTInputAdapter(2, 3, num_samples=8, temporal_kernel=3, dropout=0, max_norm=.5).double().eval()
        with torch.no_grad():
            adapter.chan_conv[0].weight.fill_(2)
        state = {name: parameter.detach().clone().requires_grad_(True) for name, parameter in adapter.named_parameters()}
        state['chan_conv.0.weight'] = torch.renorm(state['chan_conv.0.weight'].detach(), p=2, dim=0, maxnorm=.5).requires_grad_(True)
        x = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        y = (other @ torch.tensor(resample(np.eye(5), 8, axis=-1))).unsqueeze(2)
        y = F.conv2d(y, state['chan_conv.0.weight'], state['chan_conv.0.bias'])
        y = F.gelu(F.batch_norm(y, adapter.chan_conv[1].running_mean.clone(), adapter.chan_conv[1].running_var.clone(),
                               state['chan_conv.1.weight'], state['chan_conv.1.bias'], training=False, eps=1e-5))
        y = F.conv2d(y, state['chan_conv.3.weight'], state['chan_conv.3.bias'], padding=(0, 1), groups=3)
        expected = F.gelu(F.batch_norm(y, adapter.chan_conv[4].running_mean.clone(), adapter.chan_conv[4].running_var.clone(),
                                      state['chan_conv.4.weight'], state['chan_conv.4.bias'], training=False, eps=1e-5)).squeeze(2)
        actual = adapter(x)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-12)
        for name, parameter in adapter.named_parameters():
            torch.testing.assert_close(parameter.grad, state[name].grad, rtol=1e-9, atol=1e-12)

    def test_train_through_frozen_encoder_and_keep_test_statistics_fixed(self):
        encoder = self.encoder()
        snapshot = {name: value.clone() for name, value in encoder.state_dict().items()}
        adapter = EEGPTInputAdapter(2, 3, num_samples=128, temporal_kernel=3)
        model = nn.Sequential(adapter, EEGPTClassifier(encoder, freeze_encoder=True, dropout=0))
        model.train()
        self.assertTrue(adapter.training)
        self.assertFalse(encoder.training)
        before = adapter.chan_conv[3].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 2, 101, requires_grad=True)
        F.cross_entropy(model(x), torch.tensor([0, 1, 0, 1])).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, adapter.chan_conv[3].weight))
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        statistics = [(module, module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
                      for module in adapter.modules() if isinstance(module, nn.BatchNorm2d)]
        model.eval()
        model(torch.randn(2, 2, 101))
        for module, mean, variance, count in statistics:
            torch.testing.assert_close(module.running_mean, mean, rtol=0, atol=0)
            torch.testing.assert_close(module.running_var, variance, rtol=0, atol=0)
            torch.testing.assert_close(module.num_batches_tracked, count)

    def test_constraints_keep_parameter_owners_across_multiple_forwards(self):
        adapter = EEGPTInputAdapter(2, 3, num_samples=16, temporal_kernel=3, dropout=0)
        parameter = adapter.chan_conv[0].weight
        pointer = parameter.data_ptr()
        with torch.no_grad():
            parameter.fill_(20)
        x = torch.randn(2, 2, 13, requires_grad=True)
        (adapter(x) + adapter(x * .8)).square().mean().backward()
        self.assertIs(adapter.chan_conv[0].weight, parameter)
        self.assertEqual(parameter.data_ptr(), pointer)
        self.assertTrue(torch.all(parameter.detach().flatten(1).norm(dim=1) <= 1.00001))
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_unconstrained_and_frozen_projection(self):
        plain = EEGPTInputAdapter(2, 3, num_samples=16, temporal_kernel=3, max_norm=None).eval()
        with torch.no_grad():
            plain.chan_conv[0].weight.fill_(20)
        before = plain.chan_conv[0].weight.clone()
        plain(torch.randn(2, 2, 13))
        torch.testing.assert_close(plain.chan_conv[0].weight, before, rtol=0, atol=0)
        adapter = EEGPTInputAdapter(2, 3, num_samples=16, temporal_kernel=3).eval().requires_grad_(False)
        with torch.no_grad():
            adapter.chan_conv[0].weight.fill_(20)
        before = adapter.chan_conv[0].weight.clone()
        x = torch.randn(2, 2, 13, requires_grad=True)
        adapter(x).square().mean().backward()
        torch.testing.assert_close(adapter.chan_conv[0].weight, before, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_state_copy_dtype_and_eval_behavior(self):
        adapter = EEGPTInputAdapter(2, 3, num_samples=16, temporal_kernel=3, dropout=.8).double().eval()
        x = torch.randn(2, 2, 13, dtype=torch.float64)
        expected = adapter(x)
        torch.testing.assert_close(adapter(x), expected, rtol=0, atol=0)
        self.assertEqual(expected.dtype, torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            torch.save(adapter.state_dict(), path)
            restored = EEGPTInputAdapter(2, 3, num_samples=16, temporal_kernel=3, dropout=.8).double().eval()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(restored(x), expected)
        other = copy.deepcopy(adapter)
        torch.testing.assert_close(other(x), expected)
        for left, right in zip(adapter.parameters(), other.parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        adapter.float().train()
        self.assertFalse(torch.equal(adapter(x.float()), adapter(x.float())))

    def test_strict_export_resample_and_frozen_composite(self):
        x = torch.randn(2, 2, 101)
        models = [EEGPTResample(128), EEGPTInputAdapter(2, 3, 128, temporal_kernel=3),
                  EEGPTAdaptedModel(EEGPTInputAdapter(2, 3, 128, temporal_kernel=3), EEGPTClassifier(self.encoder()))]
        for model in models:
            with self.subTest(model=type(model)), tempfile.TemporaryDirectory() as directory:
                model.eval().requires_grad_(False)
                exported = torch.export.export(model, (x,), strict=True)
                path = Path(directory) / 'adapter.pt2'
                torch.export.save(exported, path)
                torch.testing.assert_close(torch.export.load(path).module()(x), model(x))

    def test_named_composition_validates_geometry_and_exposes_features(self):
        adapter = EEGPTInputAdapter(2, 3, num_samples=128)
        task = EEGPTClassifier(self.encoder(), freeze_encoder=True)
        model = EEGPTAdaptedModel(adapter, task).eval()
        x = torch.randn(2, 2, 101)
        torch.testing.assert_close(model(x), task(adapter(x)))
        torch.testing.assert_close(model.feature(x), task.feature(adapter(x)))
        self.assertEqual(model.feature(x, flatten=True).shape, (2, 128))
        model.train()
        self.assertTrue(adapter.training)
        self.assertFalse(task.encoder.training)
        for wrong_adapter, wrong_model in [(nn.Identity(), task), (adapter, nn.Linear(3, 2)),
                                          (EEGPTInputAdapter(2, 4, 128), task),
                                          (EEGPTInputAdapter(2, 3, 127), task)]:
            with self.assertRaises(ValueError):
                EEGPTAdaptedModel(wrong_adapter, wrong_model)

    def test_invalid_constructor_and_shapes(self):
        cases = [dict(in_channels=0), dict(out_channels=True), dict(num_samples=-1),
                 dict(temporal_kernel=2), dict(dropout=1), dict(max_norm=-1), dict(max_norm=float('nan'))]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                EEGPTInputAdapter(**dict(dict(in_channels=2, out_channels=3), **parameters))
        model = EEGPTInputAdapter(2, 3)
        for x in [None, torch.randn(2, 3, 10), torch.randn(2, 2), torch.empty(2, 2, 0)]:
            with self.subTest(type=type(x)), self.assertRaises(ValueError):
                model(x)


if __name__ == '__main__':
    unittest.main()
