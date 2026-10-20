import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from torcheeg.models import BIOT, BIOTClassifier
from torcheeg.models.transformer.biot import _LinearAttention
from test.models.test_eegpt import reference_state


def controlled_dropout(module, x):
    if not module.training or module.p == 0:
        return x
    # Fixed Bernoulli masks isolate dropout placement/scaling from differences
    # in the random number kernels used by supported PyTorch versions.
    keep = (torch.arange(x.numel(), device=x.device).reshape(x.shape) % 10).to(x.dtype) >= module.p * 10
    return x * keep / (1 - module.p)


class TestBIOT(unittest.TestCase):
    def model(self, **kwargs):
        config = dict(hid_channels=16, num_heads=4, depth=2, num_electrodes=4, n_fft=20, hop_length=10)
        config.update(kwargs)
        return BIOT(**config)

    def assert_summary(self, value, expected):
        if expected is None:
            self.assertIsNone(value)
            return
        self.assertIsNotNone(value)
        flat = value.detach().flatten()
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(flat.sum(), torch.tensor(expected['sum'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(flat.norm(), torch.tensor(expected['norm'], dtype=torch.float64), rtol=1e-7, atol=1e-9)

    def test_official_reference_outputs_and_all_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/biot_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'd138e32634e52ae9fa6ec98ac9c4087b14ca869a')
        self.assertEqual(fixture['attention_version'], '0.19.1')
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            for case in fixture['cases']:
                with self.subTest(mode=case['mode']):
                    model = BIOT().double()
                    model.train(case['mode'] == 'train')
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if name not in ('index', 'positional_encoding.pe'):
                                value.copy_(reference_state(name, value))
                        model.positional_encoding.pe[:, :3].copy_(torch.tensor(fixture['position_prefix'], dtype=torch.float64))
                    values = torch.arange(2 * 2 * 400, dtype=torch.float64)
                    x = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(2, 2, 400)
                    if case['mode'] == 'zero':
                        x.zero_()
                    x.requires_grad_()
                    torch.manual_seed(913)
                    with patch.object(nn.Dropout, 'forward', controlled_dropout):
                        y = model(x, channel_ids=[2, 3] if case['mode'] == 'offset' else None)
                    torch.testing.assert_close(y, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-9, atol=1e-9)
                    coefficients = torch.sin(torch.arange(y.numel(), dtype=torch.float64) / 13).reshape(y.shape)
                    (y * coefficients).square().mean().backward()
                    self.assert_summary(x.grad, case['input_gradient'])
                    self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                    for name, parameter in model.named_parameters():
                        with self.subTest(parameter=name):
                            self.assert_summary(parameter.grad, case['parameter_gradients'][name])
        finally:
            torch.set_num_threads(previous)

    def test_native_random_dropout_training_and_eval(self):
        model = self.model().train()
        x = torch.randn(2, 3, 80)
        torch.manual_seed(47)
        first = model(x)
        torch.manual_seed(47)
        torch.testing.assert_close(first, model(x), rtol=0, atol=0)
        self.assertFalse(torch.equal(first, model(x)))
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
        self.assertFalse(torch.equal(first, model(x)))

    def test_linear_attention_matches_explicit_factorization(self):
        attention = _LinearAttention(16, 4, 0.).double().eval()
        x = torch.randn(2, 11, 16, dtype=torch.float64)
        q, k, v = [layer(x).reshape(2, 11, 4, 4).transpose(1, 2)
                   for layer in (attention.to_q, attention.to_k, attention.to_v)]
        expected = (q.softmax(-1) / 2) @ (k.softmax(-2).transpose(-1, -2) @ v)
        expected = attention.to_out(expected.transpose(1, 2).reshape(2, 11, 16))
        torch.testing.assert_close(attention(x), expected)

    def test_rectangular_stft_has_no_implicit_center_log_or_normalization(self):
        model = self.model().double().eval()
        x = torch.randn(2, 1, 83, dtype=torch.float64)
        frames = x[:, 0].unfold(-1, 20, 10)
        spectrum = torch.fft.rfft(frames).abs()
        tokens = model.patch_embedding.projection(spectrum) + model.channel_tokens.weight[0]
        expected = model.transformer(model.positional_encoding(tokens)).mean(1)
        torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)

    def test_reordered_channel_ids_preserve_features_and_gradients(self):
        model = self.model().double().eval()
        x = torch.randn(2, 3, 80, dtype=torch.float64, requires_grad=True)
        order = [2, 0, 1]
        y = model(x, channel_ids=[0, 1, 3])
        z = model(x[:, order], channel_ids=[3, 0, 1])
        torch.testing.assert_close(y, z)
        self.assertFalse(torch.allclose(y, model(x[:, order], channel_ids=[0, 1, 3])))
        gy, = torch.autograd.grad(y.square().mean(), x, retain_graph=True)
        gz, = torch.autograd.grad(z.square().mean(), x)
        torch.testing.assert_close(gy, gz)

    def test_zero_input_has_finite_gradients(self):
        model = self.model()
        x = torch.zeros(2, 3, 80, requires_grad=True)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_variable_lengths_and_native_frame_limit(self):
        model = self.model().eval()
        for length in (20, 21, 80, 101):
            self.assertEqual(model(torch.randn(2, 3, length)).shape, (2, 16))
        self.assertEqual(model(torch.randn(1, 2, 10010)).shape, (1, 16))  # 1000 frames, >1024 total tokens
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 10020))

    def test_bad_architectures_inputs_and_ids(self):
        for kwargs in [dict(hid_channels=15), dict(hid_channels=18), dict(depth=0), dict(num_heads=True),
                       dict(n_fft=0), dict(hop_length=-1), dict(dropout=1), dict(attn_dropout=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)
        model = self.model()
        for x in [None, torch.ones(2, 3, 80, dtype=torch.long), torch.randn(2, 3, 19),
                  torch.randn(0, 3, 80), torch.randn(2, 0, 80), torch.randn(2, 5, 80), torch.randn(2, 1, 3, 80)]:
            with self.subTest(x=None if x is None else x.shape), self.assertRaises(ValueError):
                model(x)
        for ids in [[0], [0, 1, 4], [0, 0, 1], [-1, 0, 1], [False, 1, 2], [0., 1, 2], torch.arange(3)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                model(torch.randn(2, 3, 80), channel_ids=ids)

    def test_classifier_optimization_and_frozen_dropout(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = BIOTClassifier(self.model(), num_classes=3, freeze_backbone=frozen).train()
                before = {k: v.detach().clone() for k, v in model.named_parameters()}
                self.assertEqual(model.biot.training, not frozen)
                x = torch.randn(2, 3, 80)
                if frozen:
                    torch.testing.assert_close(model.biot(x), model.biot(x))
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 2]))
                loss.backward()
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if name.startswith('biot.') and frozen:
                        self.assertIsNone(parameter.grad)
                        self.assertTrue(torch.equal(before[name], parameter))
                    else:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertFalse(torch.equal(before['classifier.1.weight'], model.classifier[1].weight))
                if not frozen:
                    for name in ['biot.patch_embedding.projection.weight', 'biot.transformer.layers.layers.0.0.fn.to_q.weight', 'biot.channel_tokens.weight']:
                        self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]))

    def test_complete_classifier_roundtrip(self):
        model = BIOTClassifier(self.model(), 3).eval()
        other = BIOTClassifier(self.model(), 3).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            other.load_state_dict(torch.load(path, weights_only=True), strict=True)
        x = torch.randn(2, 3, 80)
        torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)

    def test_strict_pretrained_selection_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.pt'
            for variant, channels in BIOT.variants.items():
                original = BIOT(num_electrodes=channels).eval()
                state = original.state_dict()
                torch.save(state, path)
                restored = BIOT.from_pretrained(path, variant=variant).eval()
                x = torch.randn(1, 3, 200)
                torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)
            valid = BIOT().state_dict()
            prefixed = {'encoder.' + k: v for k, v in valid.items()}
            prefixed['head.weight'] = torch.zeros(2, 256)
            torch.save({'state_dict': prefixed}, path)
            BIOT.from_pretrained(path, prefix='encoder.')
            corruptions = []
            state = dict(valid); state.pop('channel_tokens.weight'); corruptions.append(state)
            state = dict(valid); state['extra'] = torch.ones(1); corruptions.append(state)
            state = dict(valid); state['index'] = valid['index'].flip(0); corruptions.append(state)
            state = dict(valid); state['index'] = valid['index'].float(); corruptions.append(state)
            state = dict(valid); state['channel_tokens.weight'] = torch.zeros(16, 256); corruptions.append(state)
            state = dict(valid); state['patch_embedding.projection.bias'] = torch.full((256,), float('nan')); corruptions.append(state)
            state = dict(valid); state['positional_encoding.pe'] = valid['positional_encoding.pe'] + .01; corruptions.append(state)
            corruptions.extend([[], {'state_dict': []}, {1: torch.ones(1)}, {'state_dict': {'x': 'not a tensor'}}])
            for state in corruptions:
                torch.save(state, path)
                with self.subTest(kind=str(type(state))), self.assertRaises(ValueError):
                    BIOT.from_pretrained(path)
            with self.assertRaises(ValueError):
                BIOT.from_pretrained(path, variant='invalid')
            with self.assertRaises(ValueError):
                BIOT.from_pretrained(path, prefix=None)

    def test_invalid_classifier_arguments(self):
        for kwargs in [dict(backbone=nn.Identity()), dict(backbone=self.model(), num_classes=0),
                       dict(backbone=self.model(), freeze_backbone=1)]:
            with self.assertRaises(ValueError):
                BIOTClassifier(**kwargs)


if __name__ == '__main__':
    unittest.main()
