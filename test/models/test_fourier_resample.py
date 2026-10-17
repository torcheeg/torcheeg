import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import resample
import torch
from torch import nn

from torcheeg.models import FourierResample, EEGPTResample, EEGPTInputAdapter
from torcheeg.models.layers import FourierResample as LayerImport
from torcheeg.models.transformer.eegpt_input import EEGPTResample as LegacyImport


class TestFourierResample(unittest.TestCase):
    def test_public_imports_legacy_type_default_and_empty_state(self):
        self.assertIs(FourierResample, LayerImport)
        self.assertIs(EEGPTResample, LegacyImport)
        legacy = EEGPTResample()
        self.assertIsInstance(legacy, FourierResample)
        self.assertEqual(legacy.num_samples, 1024)
        self.assertEqual(type(legacy).__module__, 'torcheeg.models.transformer.eegpt_input')
        self.assertEqual(type(legacy).__name__, 'EEGPTResample')
        for model in [legacy, FourierResample(200)]:
            self.assertEqual(dict(model.state_dict()), {})
            self.assertEqual(list(model.parameters()), [])
        with self.assertRaises(TypeError):
            FourierResample()

    def test_legacy_outputs_and_gradients_match_across_ranks_and_dtypes(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for shape, target in [((7,), 12), ((2, 8), 4), ((2, 3, 5), 8), ((2, 3, 4, 8), 16)]:
                with self.subTest(dtype=dtype, shape=shape, target=target):
                    x = torch.randn(shape, dtype=dtype, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    actual, expected = FourierResample(target)(x), EEGPTResample(target)(y)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual.float().square().mean().backward()
                    expected.float().square().mean().backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device, x.device)

    def test_patchwise_physical_frequency_and_amplitude(self):
        # Four one-second segments at 128 Hz become four one-second segments
        # at 200 Hz; keep the 10 Hz oscillation and per-channel scale.
        values = torch.cos(2 * torch.pi * 10 * torch.arange(128, dtype=torch.float64) / 128)
        scale = torch.arange(1, 25, dtype=torch.float64).reshape(2, 3, 4, 1)
        source = values * scale
        actual = FourierResample(200)(source)
        expected = torch.cos(2 * torch.pi * 10 * torch.arange(200, dtype=torch.float64) / 200) * scale
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual.numpy(), resample(source.numpy(), 200, axis=-1), rtol=1e-12, atol=1e-12)

    def test_broadcast_views_and_gradients_match_independent_linear_operator(self):
        source = torch.randn(1, 1, 4, 7, dtype=torch.float64, requires_grad=True)
        reference = source.detach().clone().requires_grad_()
        expanded = source.expand(2, 3, -1, -1)
        self.assertEqual(expanded.stride()[:2], (0, 0))
        actual = FourierResample(13)(expanded)
        matrix = torch.tensor(resample(np.eye(7), 13, axis=-1))
        expected = reference.expand(2, 3, -1, -1) @ matrix
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(source.grad, reference.grad, rtol=1e-10, atol=1e-12)

    def test_whole_window_and_patchwise_resampling_are_distinct(self):
        source = torch.linspace(-1, 1, 512, dtype=torch.float64).reshape(1, 1, 512)
        whole = FourierResample(800)(source)
        patches = FourierResample(200)(source.reshape(1, 1, 4, 128)).reshape(1, 1, 800)
        self.assertGreater((whole - patches).abs().max().item(), .01)
        np.testing.assert_allclose(whole.numpy(), resample(source.numpy(), 800, axis=-1), rtol=1e-12, atol=1e-12)
        expected = resample(source.numpy().reshape(1, 1, 4, 128), 200, axis=-1).reshape(1, 1, 800)
        np.testing.assert_allclose(patches.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_segments_are_isolated_in_outputs_and_gradients(self):
        x = torch.randn(2, 3, 4, 8, dtype=torch.float64, requires_grad=True)
        layer = FourierResample(16)
        expected = layer(x)
        changed = x.detach().clone()
        changed[:, :, 0].add_(10)
        actual = layer(changed)
        torch.testing.assert_close(actual[:, :, 1:], expected[:, :, 1:], rtol=0, atol=0)
        expected[:, :, 1].square().mean().backward()
        torch.testing.assert_close(x.grad[:, :, [0, 2, 3]], torch.zeros_like(x.grad[:, :, [0, 2, 3]]), rtol=0, atol=0)
        self.assertGreater(x.grad[:, :, 1].norm().item(), 0)

    def test_nyquist_corrections_on_independent_segments(self):
        for target, frequency in [(16, 4), (4, 2)]:
            source = torch.cos(2 * torch.pi * frequency * torch.arange(8, dtype=torch.float64) / 8)
            source = source.expand(2, 3, 4, -1)
            actual = FourierResample(target)(source)
            expected = torch.cos(2 * torch.pi * frequency * torch.arange(target, dtype=torch.float64) / target).expand_as(actual)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_state_restoration_copy_and_bandlimited_roundtrip(self):
        t = torch.arange(128, dtype=torch.float64) / 128
        x = (torch.sin(2 * torch.pi * 7 * t) + .5 * torch.cos(2 * torch.pi * 12 * t)).expand(2, 3, 4, -1)
        model = nn.Sequential(FourierResample(200), EEGPTResample(128))
        torch.testing.assert_close(model(x), x, rtol=1e-12, atol=1e-12)
        copied = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layers.pt'
            torch.save(dict(state_dict=model.state_dict(), num_samples=[200, 128]), path)
            payload = torch.load(path, weights_only=True)
            restored = nn.Sequential(FourierResample(payload['num_samples'][0]), EEGPTResample(payload['num_samples'][1]))
            restored.load_state_dict(payload['state_dict'], strict=True)
        torch.testing.assert_close(copied(x), restored(x), rtol=0, atol=0)
        self.assertIsInstance(EEGPTInputAdapter(19, 3).resample, EEGPTResample)
        self.assertIsInstance(EEGPTInputAdapter(19, 3).resample, FourierResample)

    def test_export_save_load_for_both_public_types(self):
        x = torch.randn(2, 3, 4, 128)
        for kind in (FourierResample, EEGPTResample):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                model = kind(200).eval()
                exported = torch.export.export(model, (x,), strict=True)
                path = Path(directory) / 'resample.pt2'
                torch.export.save(exported, path)
                restored = torch.export.load(path).module()
                torch.testing.assert_close(restored(x), model(x))

    def test_identity_and_random_state_are_preserved(self):
        x = torch.randn(2, 3, 4, 128)
        state = torch.random.get_rng_state().clone()
        layer = FourierResample(128)
        self.assertIs(layer(x), x)
        FourierResample(200)(x)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

    def test_invalid_configuration_and_inputs_match_legacy(self):
        for target in [0, -1, True, 1.5]:
            for kind in (FourierResample, EEGPTResample):
                with self.subTest(kind=kind, target=target), self.assertRaises(ValueError):
                    kind(target)
        for value in [None, [], torch.ones(()), torch.empty(0, 8), torch.ones(8, dtype=torch.long), torch.ones(8, dtype=torch.complex64)]:
            for kind in (FourierResample, EEGPTResample):
                with self.subTest(kind=kind, value_type=type(value)), self.assertRaises(ValueError):
                    kind(200)(value)


if __name__ == '__main__':
    unittest.main()
