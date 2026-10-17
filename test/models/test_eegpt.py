import json
from pathlib import Path
import pickle
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import EEGPT


def reference_state(name, value):
    count = value.numel()
    if 'norm' in name and name.endswith('weight'):
        flat = torch.linspace(0.8, 1.2, count, dtype=torch.float64)
    elif name.endswith('bias'):
        flat = torch.linspace(-0.02, 0.03, count, dtype=torch.float64)
    else:
        offset = (sum(map(ord, name)) % 11 - 5) * 0.002
        flat = torch.linspace(-0.08, 0.08, count, dtype=torch.float64) + offset
    return flat.reshape(value.shape)


class TestEEGPT(unittest.TestCase):
    def config(self, **kwargs):
        parameters = dict(num_electrodes=3, chunk_size=128, patch_size=32,
                          hid_channels=16, embed_num=2, depth=2, num_heads=4,
                          mlp_ratio=2.0, channel_names=['C3', 'CZ', 'C4'])
        parameters.update(kwargs)
        return parameters

    def model(self, **kwargs):
        return EEGPT(**self.config(**kwargs))

    def assert_summary(self, value, expected):
        flat = value.detach().reshape(-1)
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64),
                                   rtol=1e-7, atol=1e-9)
        for actual, target in [(flat.sum(), expected['sum']), (flat.norm(), expected['norm'])]:
            torch.testing.assert_close(actual, torch.tensor(target, dtype=torch.float64),
                                       rtol=1e-7, atol=1e-9)

    def test_independent_official_encoder_logits_and_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'eegpt_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'ea7282f698c406cd423ccbedbe9960853761144c')
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(fixture['threads'])
            for case in fixture['cases']:
                with self.subTest(parameters=case['parameters'], mask_x=case['mask_x'], mask_t=case['mask_t']):
                    model = EEGPT(**case['parameters'], channel_names=case['channel_names']).double().eval()
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            value.copy_(reference_state(name, value))
                    values = torch.arange(2 * model.num_electrodes * model.chunk_size, dtype=torch.float64)
                    eeg = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(
                        2, model.num_electrodes, model.chunk_size).requires_grad_(True)
                    masks = {key: torch.tensor(case[key]) if case[key] is not None else None
                             for key in ('mask_x', 'mask_t')}
                    output = model(eeg, **masks)
                    torch.testing.assert_close(output, torch.tensor(case['output'], dtype=torch.float64),
                                               rtol=1e-9, atol=1e-9)
                    coefficient = torch.sin(torch.arange(output.numel(), dtype=torch.float64) / 13).reshape(output.shape)
                    (output * coefficient).square().mean().backward()
                    self.assert_summary(eeg.grad, case['input_gradient'])
                    self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                    for name, parameter in model.named_parameters():
                        with self.subTest(parameter=name):
                            self.assert_summary(parameter.grad, case['parameter_gradients'][name])
        finally:
            torch.set_num_threads(previous)

    def test_shapes_overlap_and_patch_normalization(self):
        for kwargs, patches in [(dict(), 4), (dict(chunk_size=160, patch_stride=16), 9),
                                (dict(patch_norm=True), 4),
                                (dict(patch_norm=True, chunk_size=160, patch_stride=16), 9)]:
            with self.subTest(kwargs=kwargs):
                model = self.model(**kwargs).eval()
                eeg = torch.randn(2, 3, model.chunk_size)
                self.assertEqual(model(eeg).shape, (2, patches, 2, 16))
                if kwargs.get('patch_norm'):
                    windows = eeg.unfold(-1, 32, model.patch_stride).transpose(1, 2)
                    expected = F.linear(F.layer_norm(windows, (32,)), model.patch_embed.proj.weight,
                                        model.patch_embed.proj.bias)
                    torch.testing.assert_close(model.patch_embed(eeg), expected)

    def test_channel_vocabulary_and_explicit_ids(self):
        ids = EEGPT.prepare_chan_ids(['c3', '.cz.', ' C4 '])
        self.assertEqual(ids.tolist(), [[27, 29, 31]])
        model = self.model(channel_names=None).eval()
        eeg = torch.randn(2, 3, 128)
        with self.assertRaisesRegex(ValueError, 'channel_names'):
            model(eeg)
        torch.testing.assert_close(model(eeg, ids), model(eeg, ids.reshape(-1)))
        for names in ['C3', [], ['C3', 'c3'], ['T3'], [42], ['C3', 'bad']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                EEGPT.prepare_chan_ids(names)
        with self.assertRaises(ValueError):
            self.model(channel_names=['C3'])

    def test_channel_permutation_preserves_spatial_summary(self):
        model = self.model().eval()
        eeg = torch.randn(2, 3, 128)
        order = torch.tensor([2, 0, 1])
        original = model(eeg)
        permuted = model(eeg[:, order], model.channel_ids[:, order])
        torch.testing.assert_close(original, permuted)
        self.assertFalse(torch.allclose(original, model(eeg[:, order])))

    def test_temporal_and_grid_masks_keep_correct_positions(self):
        model = self.model().eval()
        eeg = torch.randn(2, 3, 128)
        original = model(eeg)
        torch.testing.assert_close(model(eeg, mask_t=torch.tensor([3, 1])), original[:, [3, 1]])
        selected = torch.tensor([[9, 10, 11], [3, 4, 5]])
        torch.testing.assert_close(model(eeg, mask_x=selected), original[:, [3, 1]])
        torch.testing.assert_close(model(eeg, mask_x=selected, mask_t=torch.tensor([1])), original[:, [1]])
        model(eeg, mask_t=torch.tensor([3, 1])).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_invalid_inputs_and_indices_are_rejected(self):
        model = self.model()
        for shape in [(2, 3, 127), (2, 2, 128), (2, 1, 3, 128)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(shape))
        eeg = torch.randn(2, 3, 128)
        masks = [dict(mask_x=torch.tensor([0, 1])), dict(mask_x=torch.tensor([[12]])),
                 dict(mask_x=torch.tensor([[-1]])), dict(mask_t=torch.tensor([4])),
                 dict(mask_t=torch.tensor([1.])), dict(mask_t=torch.tensor([], dtype=torch.long)),
                 dict(mask_t=torch.tensor([[1]])), dict(chan_ids=torch.tensor([1, 1, 2])),
                 dict(chan_ids=torch.tensor([1, 2, 62])), dict(chan_ids=torch.tensor([1., 2., 3.])),
                 dict(chan_ids=torch.tensor([[1, 2, 3], [1, 2, 3]]))]
        for kwargs in masks:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(eeg, **kwargs)

    def test_attention_matches_manual_softmax(self):
        model = self.model(return_attention_layer=1).eval()
        eeg = torch.randn(2, 3, 128)
        x = model.patch_embed(eeg) + model.chan_embed(model.channel_ids.reshape(-1)).reshape(1, 1, 3, 16)
        x = x.reshape(8, 3, 16)
        x = torch.cat([x, model.summary_token.expand(8, -1, -1)], dim=1)
        block = model.blocks[0]
        q, k, _ = block.attn.qkv(block.norm1(x)).reshape(8, 5, 3, 4, 4).permute(2, 0, 3, 1, 4)
        expected = (q @ k.transpose(-2, -1) / 2).softmax(-1)
        with self.assertRaises(ValueError):
            model(eeg, mask_t=torch.tensor([0]))
        torch.testing.assert_close(model(eeg), expected)
        torch.testing.assert_close(model(eeg).sum(-1), torch.ones(8, 4, 5))

    def test_optimizer_updates_patch_attention_summary_and_channels(self):
        torch.manual_seed(42)
        model = self.model(dropout=0.1, attn_dropout=0.1, drop_path=0.1, patch_norm=True)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        eeg = torch.randn(2, 3, 128, requires_grad=True)
        output = model(eeg)
        coefficient = torch.linspace(0, 1, output.numel()).reshape(output.shape)
        loss = (output * coefficient).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        for name in ['patch_embed.proj.weight', 'blocks.0.attn.qkv.weight', 'summary_token', 'chan_embed.weight']:
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        model.eval()
        torch.testing.assert_close(model(eeg), model(eeg))

    def test_zero_windows_are_finite(self):
        model = self.model()
        eeg = torch.zeros(2, 3, 128, requires_grad=True)
        model(eeg).square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_safe_checkpoint_roundtrip_and_prefix(self):
        model = self.model().eval()
        eeg = torch.randn(2, 3, 128)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'encoder.pt'
            state = model.state_dict()
            torch.save(state, checkpoint)
            loaded = EEGPT.from_pretrained(checkpoint, prefix='', **self.config()).eval()
            torch.testing.assert_close(loaded(eeg), model(eeg))
            prefixed = {'target_encoder.' + key: value for key, value in state.items()}
            prefixed['predictor.unrelated'] = torch.ones(3)
            torch.save({'state_dict': prefixed, 'epoch': 1}, checkpoint)
            loaded = EEGPT.from_pretrained(checkpoint, **self.config()).eval()
            torch.testing.assert_close(loaded(eeg), model(eeg))
            self.assertNotIn('channel_ids', loaded.state_dict())

    def test_checkpoint_rejects_incomplete_or_malformed_encoder(self):
        model = self.model()
        state = model.state_dict()
        missing = dict(state)
        missing.pop('summary_token')
        cases = [dict(), missing, dict(state, unexpected=torch.ones(1)),
                 dict(state, summary_token=torch.zeros(1)),
                 dict(state, summary_token='invalid'),
                 dict(state, summary_token=torch.full_like(state['summary_token'], float('nan'))),
                 dict(state, summary_token=torch.zeros_like(state['summary_token'], dtype=torch.long)),
                 {'state_dict': []}, []]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'encoder.pt'
            for data in cases:
                with self.subTest(data_type=type(data)):
                    torch.save(data, checkpoint)
                    with self.assertRaises(ValueError):
                        EEGPT.from_pretrained(checkpoint, prefix='', **self.config())
            torch.save({'state_dict': state, 'object': nn.Linear(2, 2)}, checkpoint)
            with self.assertRaises(pickle.UnpicklingError):
                EEGPT.from_pretrained(checkpoint, prefix='', **self.config())

    def test_constructor_rejects_invalid_parameters(self):
        cases = [dict(num_electrodes=63), dict(chunk_size=0), dict(chunk_size=16),
                 dict(patch_stride=0), dict(hid_channels=15), dict(embed_num=True),
                 dict(num_heads=0), dict(depth=-1), dict(dropout=1), dict(attn_dropout=-0.1),
                 dict(drop_path=float('nan')), dict(mlp_ratio=0), dict(mlp_ratio=.0001),
                 dict(norm_eps=float('inf')), dict(init_std=-1), dict(patch_norm=1),
                 dict(qkv_bias=0), dict(return_attention_layer=0), dict(return_attention_layer=3)]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.model(**parameters)


if __name__ == '__main__':
    unittest.main()
