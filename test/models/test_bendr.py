import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from torcheeg.models import BENDR, BENDRContextualizer, BENDRClassifier
from torcheeg.models.transformer.bendr import _load_native_state
from test.models.test_biot import controlled_dropout
from test.models.test_eegpt import reference_state


class TinyBENDR(BENDR):
    # Exercise the complete safe factory with generated small tensor states;
    # actual official 512-wide weights are checked separately against DN3.
    def __init__(self):
        super().__init__(num_electrodes=3, hid_channels=16, mlp_channels=32, num_heads=4,
                         depth=2, encoder_dropout=.1)


def controlled_channel_dropout(module, x):
    if not module.training or module.p == 0:
        return x
    keep = (torch.arange(x.shape[0] * x.shape[1], device=x.device).reshape(x.shape[0], x.shape[1], 1) % 10).to(x.dtype) >= module.p * 10
    return x * keep / (1 - module.p)


class TestBENDR(unittest.TestCase):
    def assert_summary(self, value, expected):
        if expected is None:
            self.assertIsNone(value)
            return
        self.assertIsNotNone(value)
        flat = value.detach().flatten()
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(flat.sum(), torch.tensor(expected['sum'], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(flat.norm(), torch.tensor(expected['norm'], dtype=torch.float64), rtol=1e-7, atol=1e-9)

    def test_native_reference_outputs_and_all_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/bendr_reference.json').read_text())
        self.assertEqual(fixture['commit'], '4d477fe42d3d8ce64f3b790585bfa5c7acb84848')
        for case in fixture['cases']:
            with self.subTest(mode=case['mode']):
                model = TinyBENDR().double()
                model.train(case['mode'] == 'train')
                with torch.no_grad():
                    for prefix, module in [('encoder', model.encoder), ('contextualizer', model.contextualizer)]:
                        for name, value in module.state_dict().items():
                            value.copy_(reference_state(name, value))
                values = torch.arange(2 * 3 * 385, dtype=torch.float64)
                x = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(2, 3, 385)
                if case['mode'] == 'zero':
                    x.zero_()
                x.requires_grad_()
                mask_t, mask_c = None, None
                if case['mode'] == 'masked':
                    mask_t = torch.tensor([[True, False, False, False, True], [False, False, True, False, False]])
                    mask_c = torch.zeros(2, 16, dtype=torch.bool)
                    mask_c[0, 2] = True
                    mask_c[1, 5] = True
                torch.manual_seed(913)
                with patch.object(nn.Dropout, 'forward', controlled_dropout), \
                        patch.object(nn.Dropout1d, 'forward', controlled_channel_dropout):
                    y = model.forward_features(x, mask_t=mask_t, mask_c=mask_c)
                torch.testing.assert_close(y, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-8, atol=1e-9)
                coefficients = torch.sin(torch.arange(y.numel(), dtype=torch.float64) / 13).reshape(y.shape)
                (y * coefficients).square().mean().backward()
                self.assert_summary(x.grad, case['input_gradient'])
                self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.assert_summary(parameter.grad, case['parameter_gradients'][name])

    def test_frame_counts_and_start_token_feature(self):
        model = TinyBENDR().eval()
        for samples in (1, 95, 96, 97, 385, 1024):
            x = torch.randn(2, 3, samples)
            frames = (samples + 95) // 96
            self.assertEqual(model.encoder(x).shape, (2, 16, frames))
            full = model.forward_features(x)
            self.assertEqual(full.shape, (2, 16, frames + 1))
            torch.testing.assert_close(model(x), full[:, :, 0])

    def test_mask_order_replacement_gradients_and_no_input_mutation(self):
        context = BENDRContextualizer(16, 32, 4, 2, dropout=0).double().eval()
        x = torch.randn(2, 16, 5, dtype=torch.float64, requires_grad=True)
        original = x.detach().clone()
        mask_t = torch.zeros(2, 5, dtype=torch.bool); mask_t[0, 1] = True
        mask_c = torch.zeros(2, 16, dtype=torch.bool); mask_c[1, 4] = True
        expected = x.clone()
        expected[0, :, 1] = context.mask_replacement
        expected[1, 4] = 0
        y = context(x, mask_t, mask_c)
        torch.testing.assert_close(y, context(expected))
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad[0, :, 1].abs().sum().item(), 0)
        self.assertEqual(x.grad[1, 4].abs().sum().item(), 0)
        self.assertGreater(context.mask_replacement.grad.abs().sum().item(), 0)

    def test_zero_signal_has_finite_forward_and_gradients(self):
        model = TinyBENDR().eval()
        x = torch.zeros(2, 3, 385, requires_grad=True)
        output = model(x)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_random_training_dropout_and_layer_drop_eval(self):
        model = TinyBENDR().train()
        x = torch.randn(2, 3, 385)
        torch.manual_seed(17); y = model(x)
        torch.manual_seed(17); torch.testing.assert_close(y, model(x), rtol=0, atol=0)
        self.assertFalse(torch.equal(y, model(x)))
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
        model.contextualizer.layer_drop = .9
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)

    def test_classifier_freeze_or_full_optimization(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = BENDRClassifier(TinyBENDR(), 3, freeze_backbone=frozen).train()
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                self.assertEqual(model.backbone.training, not frozen)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 385)), torch.tensor([0, 2]))
                loss.backward(); optimizer.step()
                self.assertFalse(torch.equal(before['classifier.weight'], model.classifier.weight))
                for name, parameter in model.named_parameters():
                    if frozen and name.startswith('backbone.'):
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)
                if not frozen:
                    for name in ['backbone.encoder.encoder.Encoder_0.0.weight',
                                 'backbone.contextualizer.transformer_layers.0.self_attn.in_proj_weight',
                                 'backbone.contextualizer.output_layer.weight']:
                        self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]))

    def test_complete_safe_factory_and_model_roundtrip(self):
        model = TinyBENDR().eval()
        with tempfile.TemporaryDirectory() as directory:
            encoder, context = Path(directory) / 'encoder.pt', Path(directory) / 'context.pt'
            torch.save(model.encoder.state_dict(), encoder)
            torch.save(model.contextualizer.state_dict(), context)
            restored = TinyBENDR.from_pretrained(encoder, context).eval()
            x = torch.randn(2, 3, 385)
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
            state = model.contextualizer.state_dict()
            prefixed = {'context.' + k: v for k, v in state.items()}
            prefixed['classifier.weight'] = torch.zeros(2, 16)
            torch.save({'state_dict': prefixed}, context)
            TinyBENDR.from_pretrained(encoder, context, contextualizer_prefix='context.')
            head = BENDRClassifier(model, 3).eval()
            saved = Path(directory) / 'head.pt'
            torch.save(head.state_dict(), saved)
            other = BENDRClassifier(TinyBENDR(), 3).eval()
            other.load_state_dict(torch.load(saved, weights_only=True), strict=True)
            torch.testing.assert_close(head(x), other(x), rtol=0, atol=0)

    def test_incomplete_foreign_and_nonfinite_weights_rejected(self):
        model = TinyBENDR()
        original = model.contextualizer.state_dict()
        corruptions = []
        state = dict(original); state.pop('norm.weight'); corruptions.append(state)
        state = dict(original); state['extra'] = torch.ones(1); corruptions.append(state)
        state = dict(original); state['mask_replacement'] = torch.ones(17); corruptions.append(state)
        state = dict(original); state['mask_replacement'] = torch.ones(16, dtype=torch.int64); corruptions.append(state)
        state = dict(original); state['mask_replacement'] = torch.full((16,), float('inf')); corruptions.append(state)
        corruptions.extend([[], {'state_dict': []}, {0: torch.ones(1)}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            for state in corruptions:
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    _load_native_state(model.contextualizer, path, '')
            with self.assertRaises(ValueError):
                _load_native_state(model.contextualizer, path, None)

    def test_invalid_arguments_inputs_and_masks(self):
        for kwargs in [dict(hid_channels=15), dict(hid_channels=18), dict(depth=0), dict(num_heads=7),
                       dict(encoder_dropout=1), dict(dropout=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BENDR(**kwargs)
        with self.assertRaises(ValueError):
            BENDRContextualizer(start_token=float('inf'))
        model = TinyBENDR()
        for x in [None, torch.ones(2, 3, 100, dtype=torch.int64), torch.randn(2, 4, 100),
                  torch.randn(2, 0, 100), torch.randn(0, 3, 100), torch.randn(2, 3, 0)]:
            with self.assertRaises(ValueError):
                model(x)
        for kwargs in [dict(mask_t=torch.ones(2, 5)), dict(mask_t=torch.ones(2, 4, dtype=torch.bool)),
                       dict(mask_c=torch.zeros(2, 15, dtype=torch.bool))]:
            with self.assertRaises(ValueError):
                model(torch.randn(2, 3, 385), **kwargs)
        for kwargs in [dict(backbone=nn.Identity()), dict(backbone=model, num_classes=0),
                       dict(backbone=model, freeze_backbone=1)]:
            with self.assertRaises(ValueError):
                BENDRClassifier(**kwargs)


if __name__ == '__main__':
    unittest.main()
