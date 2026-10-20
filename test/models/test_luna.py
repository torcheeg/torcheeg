import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from torcheeg.models import LUNA, LUNAClassifier
from torcheeg.models.transformer.luna import _FrequencyEmbedding, _position_encoding
from test.models.test_biot import controlled_dropout
from test.models.test_eegpt import reference_state


class TinyLUNA(LUNA):
    variants = {'base': dict(hid_channels=16, num_queries=2, depth=2)}

    def __init__(self, **kwargs):
        config = dict(hid_channels=16, num_queries=2, depth=2)
        config.update(kwargs)
        super().__init__(**config)


class TestLUNA(unittest.TestCase):
    def positions(self, batch=2, dtype=torch.float32):
        return torch.tensor([[[.1, .2, .3], [.5, 0., .9], [.2, .8, .4]]], dtype=dtype).repeat(batch, 1, 1)

    def ids(self):
        return LUNA.get_channel_ids(['FP1', 'C3', 'PZ'])

    def assert_summary(self, value, expected):
        if expected is None:
            self.assertIsNone(value)
            return
        self.assertIsNotNone(value)
        flat = value.detach().flatten()
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(flat.sum(), torch.tensor(expected['sum'], dtype=torch.float64), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(flat.norm(), torch.tensor(expected['norm'], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    def test_author_reference_outputs_attention_and_every_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/luna_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'd4484feccf62eeec666380cb55b5d5b1d517cd71')
        for case in fixture['cases']:
            with self.subTest(mode=case['mode']):
                model = TinyLUNA().double()
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        if not name.endswith('rotary_emb.freqs') and name != 'cross_attn.temparature':
                            value.copy_(reference_state(name, value))
                model.train(case['mode'] == 'train')
                if case['mode'] == 'train':
                    for module in model.modules():
                        if isinstance(module, nn.MultiheadAttention):
                            module.dropout = 0.
                values = torch.arange(3 * 80, dtype=torch.float64)
                x = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(1, 3, 80)
                if case['mode'] == 'zero':
                    x.zero_()
                x.requires_grad_()
                positions = self.positions(1, torch.float64).requires_grad_()
                mask = None
                if case['mode'] == 'masked':
                    mask = torch.zeros_like(x, dtype=torch.bool)
                    mask[0, 0, 0] = True
                    mask[0, 2, 40:] = True
                with patch.object(nn.Dropout, 'forward', controlled_dropout):
                    y, attention = model(x, positions, self.ids(), mask=mask, return_attention=True)
                torch.testing.assert_close(y, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-8, atol=1e-9)
                torch.testing.assert_close(attention.flatten(0, 1), torch.tensor(case['attention'], dtype=torch.float64), rtol=1e-8, atol=1e-9)
                coefficients = torch.sin(torch.arange(y.numel(), dtype=torch.float64) / 13).reshape(y.shape)
                ((y * coefficients).square().mean() + attention.square().mean() * .03).backward()
                self.assert_summary(x.grad, case['input_gradient'])
                self.assert_summary(positions.grad, case['position_gradient'])
                self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.assert_summary(parameter.grad, case['parameter_gradients'][name])

    def test_heterogeneous_layout_batch_matches_individual_examples(self):
        model = TinyLUNA().double().eval()
        x = torch.randn(2, 3, 120, dtype=torch.float64, requires_grad=True)
        positions = self.positions(2, torch.float64)
        positions[1] = torch.tensor([[.7, .5, .9], [.2, .8, .3], [.9, .1, .6]])
        positions.requires_grad_()
        ids = torch.tensor([self.ids(), LUNA.get_channel_ids(['F3', 'C4', 'O2'])])
        before_x, before_pos = x.detach().clone(), positions.detach().clone()
        batch = model(x, positions, ids)
        single = torch.cat([model(x[i:i + 1], positions[i:i + 1], ids[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(batch, single, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(model.forward_features(x, positions), torch.cat([
            model.forward_features(x[i:i + 1], positions[i:i + 1]) for i in range(2)]))
        first_gradient, = torch.autograd.grad(batch[0].square().mean(), positions, retain_graph=True)
        torch.testing.assert_close(first_gradient[1], torch.zeros_like(first_gradient[1]), rtol=0, atol=0)
        batch_grad = torch.autograd.grad(batch.square().mean(), (x, positions), retain_graph=True)
        single_grad = torch.autograd.grad(single.square().mean(), (x, positions))
        for actual, expected in zip(batch_grad, single_grad):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(positions, before_pos, rtol=0, atol=0)

    def test_channel_permutation_preserves_features_and_reorders_reconstruction(self):
        model = TinyLUNA().double().eval()
        x = torch.randn(2, 3, 80, dtype=torch.float64)
        positions = self.positions(2, torch.float64)
        order = [2, 0, 1]
        ids = torch.tensor(self.ids())
        torch.testing.assert_close(model.forward_features(x[:, order], positions[:, order]), model.forward_features(x, positions))
        torch.testing.assert_close(model(x[:, order], positions[:, order], ids[order]), model(x, positions, ids)[:, order])
        self.assertFalse(torch.allclose(model.forward_features(x[:, order], positions), model.forward_features(x, positions)))

    def test_patch_masks_sample_masks_gradients_and_deterministic_eval(self):
        model = TinyLUNA().double().eval()
        x = torch.randn(2, 3, 80, dtype=torch.float64, requires_grad=True)
        positions = self.positions(2, torch.float64)
        mask = torch.zeros(2, 3, 2, dtype=torch.bool); mask[0, 1, 0] = True; mask[1, 2, 1] = True
        samples = mask.repeat_interleave(40, dim=-1)
        state = torch.random.get_rng_state().clone()
        output = model(x, positions, self.ids(), mask=mask)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        torch.testing.assert_close(output, model(x, positions, self.ids(), mask=samples), rtol=0, atol=0)
        output.square().mean().backward()
        # Native masking replaces embeddings after recording-wide GroupNorm.
        # Masked samples can still affect visible embeddings through its stats.
        self.assertGreater(x.grad[samples].abs().sum().item(), 0)
        prepared, position_features = model._prepare(x, positions, mask, 0.)
        token_mask = mask.permute(0, 2, 1).reshape(-1, 3)
        expected_tokens = model.mask_token.expand(prepared.shape[0], 3, -1) + position_features
        torch.testing.assert_close(prepared[token_mask], expected_tokens[token_mask])
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(model.mask_token.grad.abs().sum().item(), 0)
        torch.testing.assert_close(model(x, positions, self.ids(), mask=torch.zeros_like(mask)), model(x, positions, self.ids()), rtol=0, atol=0)
        torch.manual_seed(31); noisy = model(x, positions, self.ids(), mask=mask, position_noise=.02)
        torch.manual_seed(31); torch.testing.assert_close(noisy, model(x, positions, self.ids(), mask=mask, position_noise=.02), rtol=0, atol=0)
        self.assertFalse(torch.equal(noisy, output))

    def test_native_spectral_and_position_order(self):
        frequency = _FrequencyEmbedding(16, 40).double().eval()
        x = torch.randn(2, 3, 80, dtype=torch.float64)
        fft = torch.fft.rfft(x.reshape(2, 3, 2, 40), dim=-1)
        expected = frequency.frequency_to_embed(torch.cat((fft.abs(), fft.angle()), dim=-1)).flatten(1, 2)
        torch.testing.assert_close(frequency(x), expected, rtol=0, atol=0)
        positions = self.positions(1, torch.float64)
        expected = torch.stack([f(positions[..., axis] * (2. ** power))
            for power in range(2) for axis in range(3) for f in (torch.sin, torch.cos)], dim=-1)
        expected = nn.functional.pad(expected, (0, 4))
        torch.testing.assert_close(_position_encoding(positions, 16), expected)

    def test_zero_inputs_and_constant_positions_have_finite_gradients(self):
        model = TinyLUNA().eval()
        x = torch.zeros(2, 3, 80, requires_grad=True)
        positions = torch.zeros(2, 3, 3, requires_grad=True)
        output = model(x, positions, self.ids())
        output.square().mean().backward()
        for value in (output, x.grad, positions.grad):
            self.assertTrue(torch.isfinite(value).all())
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_native_random_dropout_and_stochastic_depth(self):
        model = TinyLUNA(drop_path=.2).train()
        x, positions = torch.randn(2, 3, 80), self.positions()
        self.assertTrue(all(m.dropout == .1 for m in model.cross_attn.modules() if isinstance(m, nn.MultiheadAttention)))
        torch.manual_seed(19); output = model.forward_features(x, positions)
        torch.manual_seed(19); torch.testing.assert_close(output, model.forward_features(x, positions), rtol=0, atol=0)
        self.assertFalse(torch.equal(output, model.forward_features(x, positions)))
        model.eval()
        torch.testing.assert_close(model.forward_features(x, positions), model.forward_features(x, positions), rtol=0, atol=0)

    def test_classifier_frozen_probe_and_full_finetuning(self):
        for frozen in (False, True):
            model = LUNAClassifier(TinyLUNA(), 3, freeze_backbone=frozen).train()
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            self.assertEqual(model.backbone.training, not frozen)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 80), self.positions()), torch.tensor([0, 2]))
            loss.backward(); optimizer.step()
            self.assertFalse(torch.equal(before['classifier.decoder_ffn.fc2.weight'], model.classifier.decoder_ffn.fc2.weight))
            if frozen:
                for name, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, before['backbone.' + name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            else:
                for name in ['backbone.patch_embed.proj_in.0.weight', 'backbone.cross_attn.query_embed', 'backbone.blocks.0.attn.qkv_proj.weight']:
                    self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]))

    def test_bound_coordinates_generic_trainer_and_complete_state(self):
        from torch.utils.data import DataLoader, TensorDataset
        from torcheeg.trainers import ClassifierTrainer
        positions = self.positions(1)[0]
        model = LUNAClassifier(TinyLUNA(), 2, freeze_backbone=True, positions=positions)
        before = model.classifier.decoder_ffn.fc2.weight.detach().clone()
        x, labels = torch.randn(4, 3, 80), torch.tensor([0, 1, 1, 0])
        trainer = ClassifierTrainer(model, num_classes=2, lr=.001, accelerator='cpu', devices=1, metrics=['accuracy'])
        with tempfile.TemporaryDirectory() as directory:
            trainer.fit(DataLoader(TensorDataset(x, labels), batch_size=2), None,
                max_epochs=1, logger=False, enable_checkpointing=False,
                num_sanity_val_steps=0, default_root_dir=directory, enable_progress_bar=False)
            model.eval()
            self.assertFalse(torch.equal(before, model.classifier.decoder_ffn.fc2.weight))
            torch.testing.assert_close(model(x), model(x, positions), rtol=0, atol=0)
            other = LUNAClassifier(TinyLUNA(), 2, freeze_backbone=True, positions=torch.zeros_like(positions)).eval()
            other.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(other(x), model(x), rtol=0, atol=0)
            torch.testing.assert_close(other.positions, positions, rtol=0, atol=0)
        self.assertIsNot(model.positions, positions)
        with self.assertRaises(ValueError):
            LUNAClassifier(TinyLUNA(), positions=torch.full((3, 3), float('nan')))

    def test_canonical_base_full_safe_state_roundtrip(self):
        model = LUNA().eval()
        self.assertEqual(len(model.state_dict()), 224)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'canonical.safetensors'
            save_file(model.state_dict(), str(path))
            restored = LUNA.from_pretrained(path).eval()
            x, positions = torch.randn(1, 3, 80), self.positions(1)
            torch.testing.assert_close(restored(x, positions, self.ids()), model(x, positions, self.ids()), rtol=0, atol=0)

    def test_classifier_complete_safe_tensor_state_roundtrip(self):
        model, other = LUNAClassifier(TinyLUNA(), 3).eval(), LUNAClassifier(TinyLUNA(), 3).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            other.load_state_dict(torch.load(path, weights_only=True), strict=True)
        x, positions = torch.randn(2, 3, 80), self.positions()
        torch.testing.assert_close(model(x, positions), other(x, positions), rtol=0, atol=0)

    def test_strict_loader_rejects_corruption_and_supports_explicit_prefix(self):
        model = TinyLUNA()
        state = model.state_dict()
        corruptions = []
        invalid = dict(state); invalid.pop('decoder_head.decoder_linear.fc2.weight'); corruptions.append(invalid)
        invalid = dict(state); invalid['other.weight'] = torch.zeros(1); corruptions.append(invalid)
        invalid = dict(state); invalid['mask_token'] = torch.ones(1, 1, 17); corruptions.append(invalid)
        invalid = dict(state); invalid['mask_token'] = state['mask_token'].double(); corruptions.append(invalid)
        invalid = dict(state); invalid['mask_token'] = torch.full_like(state['mask_token'], float('nan')); corruptions.append(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            for invalid in corruptions:
                save_file(invalid, str(path))
                with self.assertRaises(ValueError):
                    TinyLUNA.from_pretrained(path)
            prefixed = {'encoder.' + key: value for key, value in state.items()}
            prefixed['classifier.weight'] = torch.zeros(3, 32)
            save_file(prefixed, str(path))
            TinyLUNA.from_pretrained(path, prefix='encoder.')
            with self.assertRaises(ValueError):
                TinyLUNA.from_pretrained(path, prefix='wrong.')
            with self.assertRaises(ValueError):
                LUNA.from_pretrained(path, variant='unknown')
            with self.assertRaises(ValueError):
                LUNA.from_pretrained(path, prefix=None)
        original_import = builtins.__import__
        def missing_safetensors(name, *args, **kwargs):
            if name == 'safetensors.torch':
                raise ImportError('dependency intentionally absent')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=missing_safetensors), self.assertRaisesRegex(ImportError, 'foundation'):
            LUNA.from_pretrained('not-opened.safetensors')

    def test_variable_lengths_channel_count_and_metadata_validation(self):
        model = TinyLUNA().eval()
        for channels, samples in [(1, 40), (3, 80), (4, 120)]:
            x, positions = torch.randn(2, channels, samples), torch.randn(channels, 3)
            self.assertEqual(model.forward_features(x, positions).shape, (2, samples // 40, 32))
        for kwargs in [dict(hid_channels=15), dict(num_heads=3), dict(patch_size=39), dict(patch_size=4),
                       dict(depth=0), dict(mlp_ratio=0), dict(drop_path=1)]:
            with self.assertRaises(ValueError):
                TinyLUNA(**kwargs)
        x, positions = torch.randn(2, 3, 80), self.positions()
        for bad in [torch.randn(2, 3, 81), torch.ones(2, 3, 80, dtype=torch.int64), torch.randn(0, 3, 80)]:
            with self.assertRaises(ValueError):
                model.forward_features(bad, positions)
        for bad in [torch.zeros(2, 3, 3, dtype=torch.int64), torch.randn(2, 4, 3), None]:
            with self.assertRaises(ValueError):
                model.forward_features(x, bad)
        for bad in [torch.ones(2, 3, 2), torch.ones(2, 3, 3, dtype=torch.bool)]:
            with self.assertRaises(ValueError):
                model.forward_features(x, positions, mask=bad)
        for bad in [[-1, 0, 1], [0, 1, 90], [0, 1], [0., 1., 2.]]:
            with self.assertRaises(ValueError):
                model(x, positions, bad)
        for bad in [float('inf'), -.1, True]:
            with self.assertRaises(ValueError):
                model.forward_features(x, positions, position_noise=bad)
        with self.assertRaises(ValueError):
            model.forward_features(x, positions, return_attention=1)
        with self.assertRaises(ValueError):
            LUNA.get_channel_ids(['unknown'])
        self.assertEqual(self.ids(), LUNA.get_channel_ids(['fp1', 'c3', 'pz']))


if __name__ == '__main__':
    unittest.main()
