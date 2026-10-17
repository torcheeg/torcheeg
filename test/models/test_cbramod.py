import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from torcheeg.models import CBraMod
from torcheeg.models.transformer.cbramod import _magnitude_spectrum
from test.models.test_eegpt import reference_state


class UnsupportedCheckpointObject:
    pass


class TestCBraMod(unittest.TestCase):
    def config(self, **kwargs):
        config = dict(hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
        config.update(kwargs)
        return config

    def model(self, **kwargs):
        return CBraMod(**self.config(**kwargs))

    def assert_summary(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertIsNotNone(actual)
        flat = actual.detach().reshape(-1)
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        for value, target in [(flat.sum(), expected['sum']), (flat.norm(), expected['norm'])]:
            torch.testing.assert_close(value, torch.tensor(target, dtype=torch.float64), rtol=1e-7, atol=1e-9)

    def test_unmodified_official_reconstruction_features_and_all_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'cbramod_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'b9e961003214326972c567eff390e75b0287e32a')
        self.assertEqual(fixture['source_sha256'], {'cbramod.py': 'caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14', 'criss_cross_transformer.py': '90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c'})
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(fixture['threads'])
            for case in fixture['cases']:
                with self.subTest(variant=case['variant']):
                    model = CBraMod(**case['parameters']).double().eval()
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if name != 'patch_embedding.mask_encoding':
                                value.copy_(reference_state(name, value))
                    values = torch.arange(2 * 3 * 2 * 200, dtype=torch.float64)
                    x = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(2, 3, 2, 200)
                    if case['variant'] == 'zero':
                        x.zero_()
                    x.requires_grad_()
                    mask = torch.tensor(case['mask'], dtype=torch.bool) if case['mask'] is not None else None
                    self.assert_summary(model.forward_features(x, mask), case['features'])
                    output = model(x, mask)
                    torch.testing.assert_close(output, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-9, atol=1e-9)
                    coefficients = torch.sin(torch.arange(output.numel(), dtype=torch.float64) / 13).reshape(output.shape)
                    (output * coefficients).square().mean().backward()
                    self.assert_summary(x.grad, case['input_gradient'])
                    self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                    for name, parameter in model.named_parameters():
                        with self.subTest(parameter=name):
                            self.assert_summary(parameter.grad, case['parameter_gradients'][name])
        finally:
            torch.set_num_threads(previous)

    def test_native_parameter_geometry_and_reconstruction_projection(self):
        canonical = CBraMod()
        self.assertEqual(sum(p.numel() for p in canonical.parameters()), 4924000)
        self.assertEqual(len(canonical.state_dict()), 211)
        model = self.model(out_features=7).eval()
        x = torch.randn(2, 3, 2, 200)
        features = model.forward_features(x)
        self.assertEqual(features.shape, (2, 3, 2, 16))
        torch.testing.assert_close(model.feature(x), features)
        torch.testing.assert_close(model.feature(x, flatten=True), features.flatten(1))
        with torch.no_grad():
            model.proj_out[0].weight.zero_()
            model.proj_out[0].bias.fill_(1)
        torch.testing.assert_close(model.forward_features(x), features)
        torch.testing.assert_close(model(x), torch.ones(2, 3, 2, 7))
        model(x).sum().backward()
        self.assertIsNotNone(model.proj_out[0].weight.grad)
        model.zero_grad(set_to_none=True)
        model.forward_features(x).square().mean().backward()
        self.assertIsNone(model.proj_out[0].weight.grad)

    def test_generalized_explicit_patch_and_feature_geometry(self):
        for patch_size, width in [(25, 16), (200, 32), (201, 36)]:
            with self.subTest(patch_size=patch_size):
                model = self.model(patch_size=patch_size, hid_channels=width, out_features=13)
                for channels, patches in [(1, 1), (3, 4)]:
                    x = torch.randn(2, channels, patches, patch_size)
                    self.assertEqual(model(x).shape, (2, channels, patches, 13))
                    self.assertEqual(model.forward_features(x).shape, (2, channels, patches, width))

    def test_magnitude_matches_complex_abs_at_zero_and_large_amplitudes(self):
        for dtype in (torch.float32, torch.float64):
            for scale in (0., 1., 1e20):
                with self.subTest(dtype=dtype, scale=scale):
                    values = torch.randn(2, 3, 2, 200, dtype=dtype) * scale
                    x = values.clone().requires_grad_()
                    y = values.clone().requires_grad_()
                    actual = _magnitude_spectrum(x)
                    expected = torch.fft.rfft(y, dim=-1, norm='forward').abs()
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected)
                    actual.sum().backward()
                    expected.sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    torch.testing.assert_close(x.grad, y.grad)

    def test_zero_input_has_finite_input_and_parameter_gradients(self):
        model = self.model().eval()
        x = torch.zeros(2, 3, 2, 200, requires_grad=True)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertIsNone(model.patch_embedding.mask_encoding.grad)

    def test_boolean_and_binary_masks_zero_selected_input_gradients(self):
        model = self.model().eval()
        mask = torch.tensor([[[True, False], [False, True], [False, False]],
                             [[False, True], [False, False], [True, False]]])
        x = torch.randn(2, 3, 2, 200, requires_grad=True)
        snapshot = x.detach().clone()
        output = model(x, mask)
        torch.testing.assert_close(model(x, mask.float()), output)
        torch.testing.assert_close(model(x, mask.long()), output)
        output.square().mean().backward()
        torch.testing.assert_close(x.grad[mask], torch.zeros_like(x.grad[mask]), rtol=0, atol=0)
        self.assertGreater(x.grad[~mask].norm().item(), 0)
        torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
        all_masked = model(x, torch.ones_like(mask))
        self.assertTrue(torch.isfinite(all_masked).all())

    def test_spatial_and_temporal_equivariance_without_position_convolution(self):
        model = self.model().double().eval()
        with torch.no_grad():
            model.patch_embedding.positional_encoding[0].weight.zero_()
            model.patch_embedding.positional_encoding[0].bias.zero_()
        x = torch.randn(2, 3, 4, 200, dtype=torch.float64)
        features = model.forward_features(x)
        spatial, temporal = [2, 0, 1], [3, 1, 0, 2]
        torch.testing.assert_close(model.forward_features(x[:, spatial]), features[:, spatial], rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(model.forward_features(x[:, :, temporal]), features[:, :, temporal], rtol=1e-9, atol=1e-9)

    def test_eval_determinism_and_training_dropout(self):
        model = self.model(dropout=.4)
        x = torch.randn(2, 3, 2, 200)
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(model(x), model(x)))

    def test_input_gradcheck_and_repeated_forward_backward(self):
        model = self.model(patch_size=25, hid_channels=8, num_heads=2, depth=1).double().eval()
        x = torch.randn(1, 2, 2, 25, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, (x,), fast_mode=True))
        (model(x).square().mean() + model(x).square().mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_low_precision_fft_path_and_frozen_input_gradients(self):
        model = self.model().eval().requires_grad_(False)
        x = torch.randn(2, 3, 2, 200, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(x)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.float().norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_layers_and_copied_models_have_independent_parameters(self):
        model = self.model().eval()
        first, second = model.encoder.layers
        self.assertNotEqual(first.self_attn_s.in_proj_weight.data_ptr(), second.self_attn_s.in_proj_weight.data_ptr())
        clone = copy.deepcopy(model)
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), clone.parameters())))
        clone(torch.randn(2, 3, 2, 200)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNotNone(clone.encoder.layers[0].self_attn_s.in_proj_weight.grad)

    def test_safe_tensor_state_roundtrip_and_prefixed_checkpoint(self):
        model = self.model().eval()
        x = torch.randn(2, 3, 2, 200)
        expected = model(x)
        for prefix in ['', 'backbone.']:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                state = {prefix + n: v for n, v in model.state_dict().items()}
                if prefix:
                    state['classifier.weight'] = torch.zeros(2, 16)
                    payload = {'state_dict': state}
                else:
                    payload = state
                torch.save(payload, path)
                with patch('torch.load', wraps=torch.load) as loader:
                    restored = CBraMod.from_pretrained(path, prefix=prefix, **self.config()).eval()
                loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs['weights_only'])
                self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')
                torch.testing.assert_close(restored(x), expected)

    def test_checkpoint_rejects_missing_extra_invalid_or_nonfinite_tensors(self):
        model = self.model()
        for defect in ['missing', 'extra', 'shape', 'integer', 'complex', 'non_tensor', 'nan', 'inf']:
            state = copy.deepcopy(model.state_dict())
            key = 'proj_out.0.weight'
            if defect == 'missing': del state[key]
            elif defect == 'extra': state['unexpected'] = torch.ones(1)
            elif defect == 'shape': state[key] = torch.zeros(1)
            elif defect == 'integer': state[key] = state[key].long()
            elif defect == 'complex': state[key] = state[key].to(torch.complex64)
            elif defect == 'non_tensor': state[key] = 'invalid'
            else: state[key].fill_(float(defect))
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'bad.pt'
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    CBraMod.from_pretrained(path, **self.config())

    def test_checkpoint_rejects_unsupported_pickle_and_invalid_outer_state(self):
        for value, exception in [([], ValueError), ({'state_dict': []}, ValueError),
                                 ({'state_dict': {1: torch.ones(1)}}, ValueError),
                                 ({'state_dict': {}, 'unsafe': UnsupportedCheckpointObject()}, pickle.UnpicklingError)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'bad.pt'
                torch.save(value, path)
                with patch('torch.load', wraps=torch.load) as loader, self.assertRaises(exception):
                    CBraMod.from_pretrained(path, **self.config())
                self.assertEqual(loader.call_count, 1)
                self.assertTrue(loader.call_args.kwargs['weights_only'])
        with self.assertRaises(ValueError):
            CBraMod.from_pretrained('not-read.pt', prefix=None)

    def test_invalid_constructor_geometry(self):
        for kwargs in [dict(patch_size=0), dict(hid_channels=True), dict(hid_channels=15),
                       dict(mlp_channels=0), dict(depth=0), dict(num_heads=3), dict(num_heads=0),
                       dict(out_features=-1), dict(dropout=1), dict(dropout=float('nan')),
                       dict(norm_eps=0), dict(norm_eps=float('inf')), dict(position_kernel=3),
                       dict(position_kernel=(3,)), dict(position_kernel=(2, 3)), dict(position_kernel=(True, 3)),
                       dict(patch_size=201)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)

    def test_invalid_inputs_and_masks(self):
        model = self.model()
        for x in [None, [], torch.ones(2, 3, 200), torch.ones(2, 3, 2, 199),
                  torch.ones(2, 3, 2, 200, dtype=torch.long), torch.ones(2, 0, 2, 200),
                  torch.ones(2, 3, 2, 200, dtype=torch.complex64)]:
            with self.subTest(input_type=type(x)), self.assertRaises(ValueError):
                model(x)
        x = torch.randn(2, 3, 2, 200)
        for mask in [[], torch.ones(2, 3), torch.full((2, 3, 2), 2), torch.full((2, 3, 2), float('nan')),
                     torch.zeros(2, 3, 2, dtype=torch.complex64)]:
            with self.assertRaises(ValueError):
                model(x, mask)
        with self.assertRaises(ValueError):
            model.feature(x, flatten=1)


if __name__ == '__main__':
    unittest.main()
