"""Independent protocol fixture; complete native validation is in a separate oracle.

The fixture uses public checkpoint state geometry but no author neural bodies
or weights. It cannot establish pretrained-model numerical equivalence.
"""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Namespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import MDJPT, MDJPTClassifier
from torcheeg.models.transformer.mdjpt import _patch_windows


class ProtocolContrastive(nn.Module):
    temperature = 0.07

    def forward(self, x):
        size = len(x) // 2
        loss = -(x[:size] * x[size:]).sum(-1).mean() / self.temperature
        return loss, x @ x.T, torch.arange(size), [x.new_zeros(()), x.new_zeros(())]


class ProtocolAlignment(nn.Module):
    def forward(self, domains):
        centers = torch.stack([x.mean((0, 2, 3)) for x in domains])
        return centers.square().mean()


class ProtocolBackend(nn.Module):
    """Deliberately simple grouped fixture, not an mdJPT neural approximation."""

    def __init__(self):
        super().__init__()
        native = MDJPT.PRETRAINING_CHANNELS
        cnn = Namespace(
            n_timeFilters=32,
            n_msFilters=4,
            n_channs=64,
            msFilter_timeLen=3,
            seg_att=15,
            avgPoolLen=15,
            timeSmootherLen=3,
            multiFact=1,
            dilation_array=[1, 3, 6, 12],
            stratified=['initial', 'middle1', 'middle2'],
            activ='softmax',
            temp=1.0,
            has_att=True,
            global_att=False,
            extract_mode='me',
        )
        self.cfg = Namespace(
            model=Namespace(
                encoder='MLLA',
                MLLA=Namespace(
                    patch_size=32,
                    patch_stride=6,
                    hidden_dim=128,
                    out_dim=32,
                    depth=2,
                    n_heads=8,
                    uni_channels=native,
                    cnn=cnn,
                ),
            ),
            data_0=Namespace(fs=125, timeLen=5),
            train=Namespace(
                loss=Namespace(
                    clisa_loss=True,
                    CDA_loss=True,
                    to_riem=False,
                    temp=0.07,
                    CDA_factor=0.01,
                )
            ),
        )
        self.uni_channelname = native
        self.channel_interpolate = np.stack(
            [np.roll(np.arange(60), -i) for i in range(60)]
        )
        self.MLLA = nn.Module()
        self.MLLA.encoder = nn.Module()
        self.MLLA.encoder.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        geometry = json.loads(
            (Path(__file__).parent / 'fixtures/mdjpt_state_geometry.json').read_text()
        )['states']
        for name, spec in geometry.items():
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            parent.register_parameter(
                parts[-1],
                nn.Parameter(
                    torch.full(spec['shape'], 0.5), requires_grad=spec['requires_grad']
                ),
            )
        for layer in self.MLLA.encoder.blocks:
            layer.self_attn.n_heads = 8
        self.MLLA.patch_size, self.MLLA.patch_stride = 32, 6
        for name in [
            'stratified',
            'activ',
            'temp',
            'has_att',
            'global_att',
            'extract_mode',
            'dilation_array',
        ]:
            setattr(self.cnn_encoder, name, getattr(cnn, name))
        self.clisa_loss, self.cda_loss = ProtocolContrastive(), ProtocolAlignment()
        self.save_fea = False
        self.cnn_encoder.saveFea = False
        self._patch_device_policy = 'independent_protocol_fixture'

    def channel_project(self, data, names):
        indices = {name: i for i, name in enumerate(names)}
        columns = []
        for i, name in enumerate(self.uni_channelname):
            if name in indices:
                columns.append(data[:, :, indices[name]])
            else:
                neighbors = [
                    indices[self.uni_channelname[int(v)]]
                    for v in self.channel_interpolate[i]
                    if self.uni_channelname[int(v)] in indices
                ][:3]
                columns.append(
                    torch.stack([data[:, :, j] for j in neighbors], dim=2).mean(2)
                )
        return torch.stack(columns, dim=2)

    def forward(self, x, dataset=0, returnMLLAout=False):
        size = len(x) // 2
        group_means = (
            x.reshape(2, size, 1, 60, 625).mean((1, 2, 3, 4)).repeat_interleave(size)
        )
        values = x.mean((1, 2))[:, :99] - group_means[:, None]
        feature_map = (
            values[:, None, None]
            + torch.arange(512, device=x.device)[None, :, None, None] / 512
        ) * self.uni_mlp.fc.weight.mean()
        out = (
            feature_map
            if self.save_fea
            else F.normalize(feature_map.mean(-1).squeeze(2).repeat(1, 2), dim=1)
        )
        covariance = x[:, 0, :32, :99].unsqueeze(2).expand(-1, -1, 64, -1)
        tokens = x[:, 0, :, :99].unsqueeze(-1).expand(-1, -1, -1, 32)
        return (out, covariance, tokens) if returnMLLAout else (out, covariance)


def adapter(mode='mean', names=None, **kwargs):
    return MDJPT(
        ProtocolBackend(),
        list(MDJPT.PRETRAINING_CHANNELS) if names is None else names,
        feature_mode=mode,
        **kwargs,
    )


class TestMDJPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(5, 60, 625)

    def test_independent_numpy_features_and_sliding_patch_gradients(self):
        model = adapter()
        values = self.x.numpy()
        # Each ordinary native half-group contains one window.
        expected_map = (
            values.mean(1)[:, :99, None]
            - values.mean((1, 2))[:, None, None]
            + np.arange(512)[None, None, :] / 512
        ) * 0.5
        expected = expected_map.mean(1)
        np.testing.assert_allclose(
            model(self.x).detach().numpy(), expected, rtol=1e-5, atol=1e-7
        )
        self.assertEqual(model(self.x, return_tokens=True).shape, (5, 512, 99))
        x = torch.arange(20, dtype=torch.float32).reshape(1, 20, 1).requires_grad_()
        patches = _patch_windows(x, 5, 3)
        np.testing.assert_array_equal(
            patches.detach().numpy(),
            np.stack([np.arange(i, i + 5) for i in range(0, 16, 3)])[None],
        )
        patches.sum().backward()
        counts = np.zeros(20)
        for i in range(0, 16, 3):
            counts[i : i + 5] += 1
        np.testing.assert_array_equal(x.grad.numpy().reshape(-1), counts)

    def test_physical_permutation_sample_isolation_and_native_coupling(self):
        model = adapter()
        before = model(self.x)
        changed = self.x.clone()
        changed[1] += 8
        torch.testing.assert_close(model(changed)[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close(
            torch.cat([model(v[None]) for v in self.x]), before, rtol=0, atol=0
        )
        reversed_model = adapter(names=list(reversed(MDJPT.PRETRAINING_CHANNELS)))
        torch.testing.assert_close(
            reversed_model(self.x.flip(1)), before, rtol=0, atol=0
        )
        native = model.forward_native(self.x[:4].reshape(2, 2, 60, 625))['features']
        other = model.forward_native(changed[:4].reshape(2, 2, 60, 625))['features']
        self.assertFalse(torch.equal(native[0], other[0]))
        noncontiguous = self.x.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(noncontiguous.is_contiguous())
        torch.testing.assert_close(model(noncontiguous), before, rtol=0, atol=0)

    def test_mean_de_projector_and_owned_fixed_branch(self):
        model = adapter('de')
        maps = model(self.x, return_tokens=True).detach().numpy()
        expected = 0.5 * np.log(2 * np.pi * np.e * maps.var(-1) + 1)
        np.testing.assert_allclose(
            model(self.x).detach().numpy(), expected, rtol=1e-5, atol=1e-7
        )
        projection = adapter('projector')
        output = projection(self.x)
        self.assertEqual(output.shape, (5, 1024))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(5))
        with self.assertRaises(ValueError):
            projection(self.x, return_tokens=True)
        with self.assertRaises(ValueError):
            MDJPT(
                projection.backend,
                list(MDJPT.PRETRAINING_CHANNELS),
                feature_mode='mean',
            )

    def test_interpolation_aliases_and_physical_configuration(self):
        with self.assertRaises(ValueError):
            adapter(names=['Fp1', 'T3'])
        with self.assertRaises(ValueError):
            adapter(names=['Fp1', 'T3'], allow_interpolation=True)
        model = adapter(
            names=['Fp1', 'T3'], channel_aliases={'T3': 'T7'}, allow_interpolation=True
        )
        self.assertEqual(model.projected_channel_names, ('FP1', 'T7'))
        data = torch.stack([torch.ones(625), torch.full((625,), 3.0)])[None]
        native = model.backend.channel_project(
            data[:, None], model.projected_channel_names
        )
        self.assertTrue(torch.equal(native[0, 0, 0], data[0, 0]))
        self.assertTrue(torch.equal(native[0, 0, 23], data[0, 1]))
        self.assertTrue(torch.equal(native[0, 0, 1], torch.full((625,), 2.0)))
        self.assertTrue(torch.isfinite(model(data)).all())
        for names, aliases in [
            (['T3', 'T7'], {'T3': 'T7'}),
            (['Fp1'], {'T4': 'T8'}),
            (['unknown'], {}),
            (['FP1', 'fp1'], {}),
        ]:
            with self.assertRaises(ValueError):
                adapter(names=names, channel_aliases=aliases, allow_interpolation=True)

    def test_native_objectives_explicit_pairs_domains_and_gradients(self):
        model = adapter('projector')
        values = [torch.randn(2, 3, 60, 625, requires_grad=True) for _ in range(2)]
        result = model.forward_pretraining(values)
        native = [model.forward_native(v) for v in values]
        contrastive = sum(model.backend.clisa_loss(v['projector'])[0] for v in native)
        alignment = model.backend.cda_loss([v['covariance_features'] for v in native])
        torch.testing.assert_close(result['contrastive_loss'], contrastive)
        torch.testing.assert_close(result['alignment_loss'], alignment)
        torch.testing.assert_close(result['loss'], contrastive + 0.01 * alignment)
        result['loss'].backward()
        for value in values:
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(torch.isfinite(model.backend.uni_mlp.fc.weight.grad).all())
        explicit = model.forward_pretraining(
            [
                {
                    'eeg': values[0].detach(),
                    'channel_names': list(MDJPT.PRETRAINING_CHANNELS),
                }
            ]
        )
        self.assertTrue(torch.isfinite(explicit['loss']))
        for bad in [
            [],
            [torch.randn(2, 2, 60, 625)],
            [torch.randn(6, 60, 625)],
            [{'eeg': values[0], 'channel_names': ['unknown']}],
        ]:
            with self.assertRaises(ValueError):
                model.forward_pretraining(bad)
        with self.assertRaises(ValueError):
            adapter().forward_pretraining(values)

    def test_complete_safe_loading_before_mutation(self):
        backend = ProtocolBackend()
        state = copy.deepcopy(backend.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'complete.pt'
            torch.save({'state_dict': state}, path)
            model = MDJPT.from_pretrained(
                path, backend, list(MDJPT.PRETRAINING_CHANNELS)
            )
            self.assertEqual(len(model.state_dict()), 83)
            bad = copy.deepcopy(state)
            bad.pop('cnn_encoder.timeConv2.weight')
            torch.save({'state_dict': bad}, path)
            with self.assertRaises(ValueError):
                MDJPT.from_pretrained(path, backend, list(MDJPT.PRETRAINING_CHANNELS))
            bad = copy.deepcopy(state)
            bad['uni_mlp.fc.weight'][0, 0] = float('nan')
            torch.save({'state_dict': bad}, path)
            with self.assertRaises(ValueError):
                MDJPT.from_pretrained(path, backend, list(MDJPT.PRETRAINING_CHANNELS))
            for name, value in backend.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            torch.save({'module.' + k: v for k, v in state.items()}, path)
            loaded = MDJPT.from_pretrained(
                path,
                ProtocolBackend(),
                list(MDJPT.PRETRAINING_CHANNELS),
                prefix='module.',
                state_key=None,
            )
            torch.testing.assert_close(loaded(self.x), model(self.x), rtol=0, atol=0)
            try:
                from safetensors.torch import save_file
            except ImportError:
                return
            safe = Path(directory) / 'complete.safetensors'
            save_file(state, str(safe))
            MDJPT.from_pretrained(
                safe,
                ProtocolBackend(),
                list(MDJPT.PRETRAINING_CHANNELS),
                state_key=None,
            )

    def test_fingerprint_restore_and_frozen_finetuned_task_heads(self):
        model = adapter()
        saved = copy.deepcopy(model.state_dict())
        empty = adapter().to_empty(device='cpu')
        empty.load_state_dict(saved, strict=True)
        torch.testing.assert_close(empty(self.x), model(self.x), rtol=0, atol=0)
        for changed in [
            adapter('de'),
            adapter(names=list(reversed(MDJPT.PRETRAINING_CHANNELS))),
        ]:
            with self.assertRaises(RuntimeError):
                changed.load_state_dict(saved, strict=True)
        frozen = MDJPTClassifier(model, freeze_backbone=True).train()
        self.assertFalse(model.training)
        F.cross_entropy(frozen(self.x), torch.tensor([0, 1, 0, 1, 0])).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNotNone(frozen.classifier.weight.grad)
        trained = MDJPTClassifier(adapter()).train()
        F.cross_entropy(trained(self.x), torch.tensor([0, 1, 0, 1, 0])).backward()
        self.assertIsNotNone(trained.backbone.backend.uni_mlp.fc.weight.grad)

    def test_invalid_samples_and_incomplete_native_geometry(self):
        model = adapter()
        for bad in [
            torch.zeros(1, 60, 624),
            torch.zeros(0, 60, 625),
            torch.zeros(1, 59, 625),
            torch.zeros(1, 60, 625, dtype=torch.float64),
            torch.full((1, 60, 625), float('inf')),
        ]:
            with self.assertRaises(ValueError):
                model(bad)
        for mutate in [
            lambda m: setattr(m.cfg.data_0, 'fs', 250),
            lambda m: setattr(m.MLLA.encoder.blocks[0].self_attn, 'n_heads', 4),
            lambda m: setattr(m, 'channel_interpolate', np.zeros((60, 60), dtype=int)),
            lambda m: m.cnn_encoder.timeConv2.register_parameter(
                'weight', nn.Parameter(torch.zeros(1))
            ),
        ]:
            backend = ProtocolBackend()
            mutate(backend)
            with self.assertRaises(ValueError):
                MDJPT(backend, list(MDJPT.PRETRAINING_CHANNELS))


if __name__ == '__main__':
    unittest.main()
