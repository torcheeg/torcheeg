import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from torcheeg.models import EEGPT, EEGPTPredictor, EEGPTReconstructor
from torcheeg.models.transformer.eegpt_decoder import _rotate, _TimeEmbedding
from test.models.test_eegpt import reference_state


class TestEEGPTDecoder(unittest.TestCase):
    def model(self, kind=EEGPTReconstructor, **kwargs):
        config = dict(num_patches=4, in_features=12, embed_num=2, hid_channels=16,
                      depth=2, num_heads=4, mlp_ratio=2.)
        config.update(kwargs)
        return kind(**config)

    def assert_summary(self, value, expected):
        if expected is None:
            self.assertIsNone(value)
            return
        self.assertIsNotNone(value)
        flat = value.detach().reshape(-1)
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        for actual, target in [(flat.sum(), expected['sum']), (flat.norm(), expected['norm'])]:
            torch.testing.assert_close(actual, torch.tensor(target, dtype=torch.float64), rtol=1e-7, atol=1e-9)

    def test_official_outputs_and_all_parameter_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'eegpt_decoder_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'ea7282f698c406cd423ccbedbe9960853761144c')
        self.assertEqual(fixture['source_sha256'], '0371411f4c2bfc0d1d0999432c8cea82e1d0f76d2e89ac95089adfeb59ed2233')
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(fixture['threads'])
            for case in fixture['cases']:
                with self.subTest(kind=case['kind'], variant=case['variant']):
                    kind = EEGPTReconstructor if case['kind'] == 'reconstructor' else EEGPTPredictor
                    model = kind(**case['parameters']).double().eval()
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if name != 'time_embed.freqs':
                                value.copy_(reference_state(name, value))
                    values = torch.arange(2 * 4 * 2 * model.in_features, dtype=torch.float64)
                    inputs = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(2, 4, 2, model.in_features).requires_grad_()
                    output = model(inputs, use_out_proj=case['use_out_proj'])
                    torch.testing.assert_close(output, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-9, atol=1e-9)
                    coefficients = torch.sin(torch.arange(output.numel(), dtype=torch.float64) / 13).reshape(output.shape)
                    (output * coefficients).square().mean().backward()
                    self.assert_summary(inputs.grad, case['input_gradient'])
                    self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                    for name, parameter in model.named_parameters():
                        with self.subTest(parameter=name):
                            self.assert_summary(parameter.grad, case['parameter_gradients'][name])
        finally:
            torch.set_num_threads(previous)

    def test_rotary_positions_and_partial_head_rotation(self):
        embedding = _TimeEmbedding(4, 2.)
        x = torch.zeros(1, dtype=torch.float64)
        frequencies = embedding(3, 2, x)
        expected = torch.outer(torch.tensor([0, 0, 0, .5, .5, 1, 1], dtype=torch.float64),
                               embedding.freqs.double()).repeat_interleave(2, -1)
        torch.testing.assert_close(frequencies, expected)
        values = torch.arange(42, dtype=torch.float64).reshape(7, 6)
        rotated = _rotate(values, frequencies)
        torch.testing.assert_close(rotated[:, 4:], values[:, 4:])
        torch.testing.assert_close(rotated[:, :4].square().sum(-1), values[:, :4].square().sum(-1))
        torch.testing.assert_close(rotated[0], values[0])
        expected_first = values[:, 0] * frequencies[:, 0].cos() - values[:, 1] * frequencies[:, 0].sin()
        torch.testing.assert_close(rotated[:, 0], expected_first)

    def test_no_frequency_cache_after_dtype_or_parameter_changes(self):
        embedding = _TimeEmbedding(8, 2.)
        original = embedding(4, 2, torch.zeros(1))
        self.assertEqual(list(embedding.__dict__.get('_buffers', {})), [])
        embedding.double()
        with torch.no_grad():
            embedding.freqs.mul_(2)
        actual = embedding(4, 2, torch.zeros(1, dtype=torch.float64))
        torch.testing.assert_close(actual, original.double() * 2)
        self.assertEqual(actual.dtype, torch.float64)

    def test_both_decoders_shapes_and_unused_native_parameters(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind)
            inputs = torch.randn(2, 4, 2, 12, requires_grad=True)
            hidden = model(inputs)
            self.assertEqual(hidden.shape, (2, 9, 16))
            hidden.square().mean().backward()
            self.assertIsNone(model.mask_token.grad)
            self.assertIsNone(getattr(model, 'reconstructor_proj' if kind == EEGPTReconstructor else 'predictor_proj').weight.grad)
            if kind == EEGPTReconstructor:
                self.assertIsNone(model.chan_embed.weight.grad)
            self.assertGreater(inputs.grad.norm().item(), 0)
            model.zero_grad(set_to_none=True)
            projected = model(inputs, use_out_proj=True)
            self.assertEqual(projected.shape[-1], model.output_features)
            projected.square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name not in ('mask_token', 'chan_embed.weight'):
                    self.assertIsNotNone(parameter.grad, name)

    def test_encoder_to_decoder_gradient_flow(self):
        encoder = EEGPT(num_electrodes=3, chunk_size=128, patch_size=32, hid_channels=12,
                        embed_num=2, depth=1, num_heads=3, channel_names=['C3', 'CZ', 'C4'])
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            encoder.zero_grad(set_to_none=True)
            decoder = self.model(kind)
            eeg = torch.randn(2, 3, 128, requires_grad=True)
            output = decoder(encoder(eeg))[:, 1:].mean(1)
            output.square().mean().backward()
            self.assertGreater(eeg.grad.norm().item(), 0)
            self.assertGreater(encoder.patch_embed.proj.weight.grad.norm().item(), 0)
            self.assertGreater(decoder.cls_token.grad.norm().item(), 0)

    def test_attention_probabilities_and_validation(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind, return_attention_layer=1).eval()
            weights = model(torch.randn(2, 4, 2, 12))
            self.assertEqual(weights.shape, (2, 4, 9, 9))
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 9))
            self.assertTrue((weights >= 0).all())

    def test_eval_determinism_and_training_dropout(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind, dropout=.4, attn_dropout=.3, drop_path=.2)
            inputs = torch.randn(2, 4, 2, 12)
            model.eval()
            torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)
            model.train()
            self.assertFalse(torch.equal(model(inputs), model(inputs)))

    def test_copy_and_tensor_only_state_roundtrip(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind, position_embedding=True).double().eval()
            inputs = torch.randn(2, 4, 2, 12, dtype=torch.float64)
            expected = model(inputs, use_out_proj=True)
            clone = copy.deepcopy(model)
            self.assertNotEqual(model.cls_token.data_ptr(), clone.cls_token.data_ptr())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'decoder.pt'
                torch.save(model.state_dict(), path)
                clone.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(clone(inputs, use_out_proj=True), expected)
            clone(inputs).sum().backward()
            self.assertIsNone(model.cls_token.grad)
            self.assertIsNotNone(clone.cls_token.grad)

    def test_single_token_and_empty_rotary_subspace(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            for width in (1, 2, 3, 4):
                model = self.model(kind, num_patches=1, embed_num=1, hid_channels=width, num_heads=1)
                output = model(torch.randn(2, 1, 1, 12))
                self.assertEqual(output.shape, (2, 2, width))
                self.assertTrue(torch.isfinite(output).all())

    def test_repeated_forward_single_backward_and_input_gradcheck(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind, num_patches=2, embed_num=1, depth=1, in_features=4,
                               hid_channels=4, num_heads=1).double().eval()
            inputs = torch.randn(1, 2, 1, 4, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(model, (inputs,), fast_mode=True))
            (model(inputs).square().mean() + model(inputs).mean()).backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_invalid_configuration(self):
        invalid = [dict(num_patches=0), dict(in_features=True), dict(embed_num=-1), dict(hid_channels=15),
                   dict(depth=0), dict(num_heads=0), dict(mlp_ratio=0), dict(mlp_ratio=float('nan')),
                   dict(dropout=1), dict(attn_dropout=-.1), dict(drop_path=float('inf')), dict(norm_eps=0),
                   dict(input_projection='yes'), dict(position_embedding=1), dict(qkv_bias=0),
                   dict(input_projection=False), dict(interpolate_factor=.5), dict(init_std=float('inf')),
                   dict(return_attention_layer=0), dict(return_attention_layer=3), dict(return_attention_layer=True)]
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            for kwargs in invalid:
                with self.subTest(kind=kind, kwargs=kwargs), self.assertRaises(ValueError):
                    self.model(kind, **kwargs)
        with self.assertRaises(ValueError):
            self.model(patch_size=0)

    def test_invalid_input_geometry(self):
        for kind in (EEGPTReconstructor, EEGPTPredictor):
            model = self.model(kind)
            for value in (None, [], torch.randn(2, 4, 12), torch.randn(2, 3, 2, 12),
                          torch.randn(2, 4, 3, 12), torch.randn(2, 4, 2, 16)):
                with self.subTest(kind=kind, value_type=type(value)), self.assertRaises(ValueError):
                    model(value)
            with self.assertRaises(ValueError):
                model(torch.randn(2, 4, 2, 12), use_out_proj=1)


if __name__ == '__main__':
    unittest.main()
