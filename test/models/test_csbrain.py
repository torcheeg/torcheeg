"""Independent small protocols for CSBrain mapping, masks and complete loading."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import CSBrain, CSBrainClassifier


CHANNELS = list(CSBrain.PRETRAINING_CHANNELS)


class ProtocolBackend(nn.Module):
    """An analytic test fixture, never a pretrained CSBrain replacement."""

    def __init__(self, regions=None, indices=None):
        super().__init__()
        self.brain_regions = list(
            CSBrain.PRETRAINING_REGIONS if regions is None else regions
        )
        self.sorted_indices = list(
            CSBrain.PRETRAINING_INDICES if indices is None else indices
        )
        self.area_config = {}
        start = 0
        for region in sorted(set(self.brain_regions)):
            size = self.brain_regions.count(region)
            self.area_config['region_' + str(region)] = {
                'channels': size,
                'slice': slice(start, start + size),
            }
            start += size
        self.patch_embedding = nn.Module()
        self.patch_embedding.d_model = 200
        self.patch_embedding.mask_encoding = nn.Parameter(
            torch.zeros(200), requires_grad=False
        )
        self.patch_embedding.positional_encoding = nn.Sequential(
            nn.Conv2d(200, 200, (19, 7), groups=200)
        )
        self.patch_embedding.spectral_proj = nn.Sequential(nn.Linear(101, 200))
        self.TemEmbed_kernel_sizes = [(1,), (3,), (5,)]
        self.encoder = nn.Module()
        self.encoder.num_layers = 12
        self.encoder.layers = nn.ModuleList()
        for _ in range(12):
            layer = nn.Module()
            layer.d_model, layer.nhead, layer.batch_first = 200, 8, True
            layer.activation = F.gelu
            layer.area_config = self.area_config
            layer.inter_region_attn = nn.MultiheadAttention(200, 8, batch_first=True)
            layer.inter_window_attn = nn.MultiheadAttention(200, 8, batch_first=True)
            layer.linear1 = nn.Linear(200, 800)
            layer.linear2 = nn.Linear(800, 200)
            self.encoder.layers.append(layer)
        self.proj_out = nn.Sequential(nn.Linear(200, 200))
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        x = x[:, self.sorted_indices]
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.patch_embedding.mask_encoding, x)
        offset = (
            torch.arange(x.shape[1], device=x.device, dtype=x.dtype).reshape(
                1, -1, 1, 1
            )
            / 100
        )
        features = self.dropout(x * self.gain + offset)
        return self.proj_out(features)


class TestCSBrain(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.backend = ProtocolBackend()
        self.model = CSBrain(self.backend, CHANNELS).eval()
        self.x = torch.randn(2, 19, 400)

    def test_independent_native_token_order_and_complete_projection(self):
        x = self.x.clone().requires_grad_()
        tokens = self.model(x, return_tokens=True)
        expected = (
            self.x.numpy().reshape(2, 19, 2, 200)[:, list(CSBrain.PRETRAINING_INDICES)]
            * 2
            + np.arange(19, dtype=np.float32).reshape(1, 19, 1, 1) / 100
        )
        np.testing.assert_allclose(
            tokens.detach().numpy(), expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.model(x).detach().numpy(), expected.mean((1, 2)), rtol=1e-6, atol=1e-6
        )
        projection = self.backend.proj_out[0]
        prediction = (
            expected @ projection.weight.detach().numpy().T
            + projection.bias.detach().numpy()
        )
        np.testing.assert_allclose(
            self.model.forward_reconstruction(x).detach().numpy(),
            prediction,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(
            self.model.token_channel_names,
            tuple(CHANNELS[i] for i in CSBrain.PRETRAINING_INDICES),
        )
        tokens.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(self.backend.gain.grad.abs(), 0)
        self.assertEqual(
            set(self.backend.proj_out.state_dict()), {'0.weight', '0.bias'}
        )

    def test_declared_reordering_prediction_and_eval_sample_isolation(self):
        backend = ProtocolBackend()
        backend.load_state_dict(self.backend.state_dict())
        declared = CSBrain(backend, list(reversed(CHANNELS))).eval()
        reversed_x = self.x.flip(1)
        torch.testing.assert_close(
            declared(reversed_x), self.model(self.x), rtol=0, atol=0
        )
        torch.testing.assert_close(
            declared.forward_reconstruction(reversed_x).flip(1),
            self.model.forward_reconstruction(self.x),
            rtol=0,
            atol=0,
        )
        before = self.model(self.x)
        changed = self.x.clone()
        changed[1] += 3
        torch.testing.assert_close(self.model(changed)[0], before[0], rtol=0, atol=0)
        self.assertEqual(self.model(self.x[..., ::2]).shape, (2, 200))

    def test_physical_mask_hides_target_and_preserves_target_gradient(self):
        mask = torch.zeros(2, 19, 2, dtype=torch.bool)
        mask[:, 1, 0] = True
        x = self.x.clone().requires_grad_()
        result = self.model.forward_masked(x, mask)
        self.assertTrue(result['native_mask'][:, 6, 0].all())
        torch.testing.assert_close(result['target_mask'], mask, rtol=0, atol=0)
        torch.testing.assert_close(result['hidden_mask'], mask, rtol=0, atol=0)
        changed = x.detach().clone()
        changed[:, 1, :200] += 100
        torch.testing.assert_close(
            self.model.forward_masked(changed, mask)['prediction'],
            result['prediction'],
            rtol=0,
            atol=0,
        )
        result['prediction'].sum().backward(retain_graph=True)
        torch.testing.assert_close(
            x.grad[:, 1, :200], torch.zeros_like(x.grad[:, 1, :200]), rtol=0, atol=0
        )
        x.grad = None
        result['loss'].backward()
        self.assertGreater(x.grad[:, 1, :200].abs().sum(), 0)

    def test_original_mask_convention_and_exact_selected_target_loss(self):
        mask = torch.zeros(2, 19, 2, dtype=torch.bool)
        mask[:, 1, 0] = True
        result = self.model.forward_native_masked(self.x, mask)
        self.assertTrue(result['target_mask'][:, 1, 0].all())
        self.assertTrue(result['hidden_mask'][:, 10, 0].all())
        self.assertFalse(result['hidden_mask'][:, 1, 0].any())
        expected = F.mse_loss(
            result['prediction'][mask], self.x.reshape(2, 19, 2, 200)[mask]
        )
        torch.testing.assert_close(result['loss'], expected, rtol=0, atol=0)
        self.model.train()
        before = self.backend.proj_out[0].weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.forward_native_masked(self.x, mask)['loss'].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.backend.proj_out[0].weight))

    def test_complete_safe_loading_failed_load_no_mutation_and_safetensors(self):
        state = copy.deepcopy(self.backend.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save({'module.' + k: v for k, v in state.items()}, path)
            loaded = CSBrain.from_pretrained(path, ProtocolBackend(), CHANNELS).eval()
            torch.testing.assert_close(
                loaded(self.x), self.model(self.x), rtol=0, atol=0
            )
            torch.save({'model': state}, path)
            CSBrain.from_pretrained(
                path, ProtocolBackend(), CHANNELS, state_key='model', prefix=''
            )
            variants = []
            missing = dict(state)
            missing.pop('proj_out.0.weight')
            variants.append(missing)
            extra = dict(state)
            extra['extra'] = torch.ones(1)
            variants.append(extra)
            for value in [
                torch.ones(1),
                state['proj_out.0.weight'].double(),
                torch.full_like(state['proj_out.0.weight'], float('inf')),
            ]:
                bad = dict(state)
                bad['proj_out.0.weight'] = value
                variants.append(bad)
            for bad in variants:
                torch.save(bad, path)
                backend = ProtocolBackend()
                before = copy.deepcopy(backend.state_dict())
                with self.assertRaises(ValueError):
                    CSBrain.from_pretrained(path, backend, CHANNELS, prefix='')
                for key, value in before.items():
                    torch.testing.assert_close(
                        backend.state_dict()[key], value, rtol=0, atol=0
                    )
            from safetensors.torch import save_file

            path = Path(directory) / 'weights.safetensors'
            save_file(state, str(path))
            loaded = CSBrain.from_pretrained(
                path, ProtocolBackend(), CHANNELS, prefix=''
            ).eval()
            torch.testing.assert_close(
                loaded(self.x), self.model(self.x), rtol=0, atol=0
            )

    def test_metadata_to_empty_and_explicit_different_montage(self):
        state = copy.deepcopy(self.model.state_dict())
        restored = CSBrain(ProtocolBackend(), CHANNELS).to_empty(device='cpu').eval()
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(self.x), self.model(self.x), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'channel order'):
            CSBrain(ProtocolBackend(), list(reversed(CHANNELS))).load_state_dict(
                state, strict=True
            )
        changed = ProtocolBackend()
        changed.dropout.p = 0.3
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            CSBrain(changed, CHANNELS).load_state_dict(state, strict=True)
        names = [
            'Fz',
            'FC3',
            'FC1',
            'FCz',
            'FC2',
            'FC4',
            'C5',
            'C3',
            'C1',
            'Cz',
            'C2',
            'C4',
            'C6',
            'CP3',
            'CP1',
            'CPz',
            'CP2',
            'CP4',
            'P1',
            'Pz',
            'P2',
            'POz',
        ]
        regions = [0] + [4] * 17 + [1] * 4
        indices = [0, 18, 19, 20, 21] + list(range(1, 18))
        with self.assertRaisesRegex(ValueError, 'allow_montage_transfer'):
            CSBrain(
                ProtocolBackend(regions, indices), names, backend_channel_names=names
            )
        transfer = CSBrain(
            ProtocolBackend(regions, indices),
            names,
            backend_channel_names=names,
            allow_montage_transfer=True,
        ).eval()
        self.assertEqual(transfer(torch.zeros(1, 22, 200)).shape, (1, 200))
        self.assertEqual(
            transfer.forward_reconstruction(torch.zeros(1, 22, 200)).shape,
            (1, 22, 1, 200),
        )

    def test_classifier_freeze_and_finetune(self):
        frozen = CSBrainClassifier(self.model, freeze_backbone=True).train()
        self.assertFalse(frozen.backbone.training)
        before = copy.deepcopy(frozen.backbone.state_dict())
        frozen(self.x).square().mean().backward()
        self.assertIsNotNone(frozen.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in frozen.backbone.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(
                frozen.backbone.state_dict()[key], value, rtol=0, atol=0
            )
        fine = CSBrainClassifier(CSBrain(ProtocolBackend(), CHANNELS)).train()
        fine(self.x).square().mean().backward()
        self.assertGreater(fine.backbone.backend.gain.grad.abs(), 0)

    def test_invalid_physical_masks_geometry_and_incomplete_backends(self):
        for names in [[], CHANNELS[:-1], ['C3'] * 19, CHANNELS[:-1] + ['unknown']]:
            with self.assertRaises(ValueError):
                CSBrain(ProtocolBackend(), names)
        for rate in [100, True]:
            with self.assertRaises(ValueError):
                CSBrain(ProtocolBackend(), CHANNELS, sampling_rate=rate)
        for x in [
            torch.zeros(0, 19, 200),
            torch.zeros(1, 18, 200),
            torch.zeros(1, 19, 199),
            torch.zeros(1, 19, 201),
            torch.ones(1, 19, 200, dtype=torch.int64),
            torch.full((1, 19, 200), float('nan')),
        ]:
            with self.assertRaises(ValueError):
                self.model(x)
        for mask in [
            torch.zeros(2, 19, 2, dtype=torch.bool),
            torch.ones(2, 19, 1, dtype=torch.bool),
            torch.ones(2, 19, 2),
        ]:
            with self.assertRaises(ValueError):
                self.model.forward_masked(self.x, mask)
        wrong = ProtocolBackend()
        wrong.sorted_indices[0] = wrong.sorted_indices[1]
        with self.assertRaises(ValueError):
            CSBrain(wrong, CHANNELS)
        wrong = ProtocolBackend()
        wrong.encoder.layers[0].nhead = 4
        with self.assertRaises(ValueError):
            CSBrain(wrong, CHANNELS)
        wrong = ProtocolBackend()
        wrong.proj_out = nn.Identity()
        with self.assertRaises(ValueError):
            CSBrain(wrong, CHANNELS)


if __name__ == '__main__':
    unittest.main()
