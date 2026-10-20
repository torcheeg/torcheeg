"""Independent protocol fixtures for external Uni-DMFM interface behavior."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from torcheeg.models import UniDMFM, UniDMFMClassifier


class ProtocolBackend(nn.Module):
    """Small independently defined test protocol; never an author model."""

    def __init__(self):
        super().__init__()
        self.d_model, self.patch_size = 384, 200
        self.stft_params = {'n_fft': 64, 'hop_length': 16}
        self.patch_embedding = nn.Module()
        self.patch_embedding.channel_pos_emb = nn.Parameter(torch.zeros(1, 256, 1, 384))
        self.patch_embedding.patch_pos_emb = nn.Parameter(torch.zeros(1, 1, 100, 384))
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList()
        for _ in range(12):
            layer = nn.Module()
            layer.d_model, layer.nhead, layer.group_size = 384, 12, 4
            layer.temporal_attn = nn.Identity()
            layer.temporal_attn.embed_dim = 384
            layer.temporal_attn.batch_first = True
            layer.drop_path = nn.Identity()
            self.transformer_encoder.layers.append(layer)
        self.projection = nn.Linear(200, 384)
        self.time_reconstruction_head = nn.Linear(384, 200)
        self.freq_reconstruction_head = nn.Sequential(
            nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 429)
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 256)
        )
        self.dropout = nn.Dropout(0.2)

    def forward_features(self, x):
        return self.dropout(
            self.projection(x)
            + self.patch_embedding.channel_pos_emb[:, : x.shape[1]]
            + self.patch_embedding.patch_pos_emb[:, :, : x.shape[2]]
        )

    def forward(self, x, mode='full'):
        features = self.forward_features(x)
        projection = self.contrastive_head(features.mean((1, 2)))
        if mode == 'contrastive':
            return projection
        return (
            self.time_reconstruction_head(features),
            self.freq_reconstruction_head(features),
            projection,
        )


class TestUniDMFM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.backend = ProtocolBackend()
        self.model = UniDMFM(self.backend, ['C3', 'Pz']).eval()
        self.x = torch.randn(2, 2, 600)

    def test_independent_patch_order_features_and_gradients(self):
        x = self.x.clone().requires_grad_()
        tokens = self.model(x, return_tokens=True)
        patches = x.detach().numpy().reshape(2, 2, 3, 200)
        expected = (
            patches @ self.backend.projection.weight.detach().numpy().T
            + self.backend.projection.bias.detach().numpy()
        )
        np.testing.assert_allclose(
            tokens.detach().numpy(), expected, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            self.model(x).detach().numpy(), expected.mean((1, 2)), rtol=1e-5, atol=1e-6
        )
        tokens.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum(), 0)
        self.assertGreater(self.backend.projection.weight.grad.abs().sum(), 0)
        self.assertEqual(self.model(x.transpose(0, 1)).shape, (2, 384))

    def test_all_three_heads_gradients_and_optimizer(self):
        x = self.x.clone().requires_grad_()
        result = self.model.forward_pretraining(x)
        self.assertEqual(result['time_prediction'].shape, (2, 2, 3, 200))
        self.assertEqual(result['frequency_prediction'].shape, (2, 2, 3, 429))
        torch.testing.assert_close(
            self.model.forward_contrastive(x),
            result['contrastive_projection'],
            rtol=0,
            atol=0,
        )
        before = {
            name: p.detach().clone()
            for name, p in self.backend.named_parameters()
            if 'head' in name
        }
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        sum(value.square().mean() for value in result.values()).backward()
        optimizer.step()
        for name, value in before.items():
            parameter = dict(self.backend.named_parameters())[name]
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertFalse(torch.equal(value, parameter), name)

    def test_sample_isolation_and_no_padding_contract(self):
        before = self.model(self.x)
        changed = self.x.clone()
        changed[1] += 8
        torch.testing.assert_close(self.model(changed)[0], before[0], rtol=0, atol=0)
        with self.assertRaises(TypeError):
            self.model(self.x, padding_mask=torch.zeros(2, 2, dtype=torch.bool))
        self.assertEqual(
            self.model(torch.zeros(1, 2, 200), return_tokens=True).shape, (1, 2, 1, 384)
        )

    def test_complete_safe_checkpoint_validation_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            state = copy.deepcopy(self.backend.state_dict())
            torch.save({'model': state}, path)
            loaded = UniDMFM.from_pretrained(
                path, ProtocolBackend(), ['C3', 'Pz']
            ).eval()
            torch.testing.assert_close(
                loaded(self.x), self.model(self.x), rtol=0, atol=0
            )
            torch.save({'model': {'module.' + k: v for k, v in state.items()}}, path)
            UniDMFM.from_pretrained(
                path, ProtocolBackend(), ['C3', 'Pz'], prefix='module.'
            )
            torch.save(state, path)
            UniDMFM.from_pretrained(
                path, ProtocolBackend(), ['C3', 'Pz'], state_key=None
            )
            bad_states = []
            missing = dict(state)
            missing.pop('contrastive_head.2.weight')
            bad_states.append(missing)
            extra = dict(state)
            extra['unexpected'] = torch.ones(1)
            bad_states.append(extra)
            for value in [
                torch.ones(2),
                state['projection.weight'].double(),
                torch.full_like(state['projection.weight'], float('nan')),
            ]:
                wrong = dict(state)
                wrong['projection.weight'] = value
                bad_states.append(wrong)
            for bad in bad_states:
                torch.save({'model': bad}, path)
                backend = ProtocolBackend()
                before = copy.deepcopy(backend.state_dict())
                with self.assertRaises(ValueError):
                    UniDMFM.from_pretrained(path, backend, ['C3', 'Pz'])
                for key, value in before.items():
                    torch.testing.assert_close(
                        backend.state_dict()[key], value, rtol=0, atol=0
                    )
            from safetensors.torch import save_file

            safe = Path(directory) / 'model.safetensors'
            save_file(state, str(safe))
            restored = UniDMFM.from_pretrained(
                safe, ProtocolBackend(), ['C3', 'Pz'], state_key=None
            ).eval()
            torch.testing.assert_close(
                restored(self.x), self.model(self.x), rtol=0, atol=0
            )

    def test_metadata_strict_restore_and_to_empty(self):
        state = copy.deepcopy(self.model.state_dict())
        restored = (
            UniDMFM(ProtocolBackend(), ['C3', 'Pz']).to_empty(device='cpu').eval()
        )
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(self.x), self.model(self.x), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'channel order'):
            UniDMFM(ProtocolBackend(), ['Pz', 'C3']).load_state_dict(state, strict=True)
        backend = ProtocolBackend()
        backend.transformer_encoder.layers[1].drop_path = nn.Module()
        backend.transformer_encoder.layers[1].drop_path.drop_prob = 0.1
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            UniDMFM(backend, ['C3', 'Pz']).load_state_dict(state, strict=True)

    def test_classifier_freeze_and_finetune(self):
        frozen = UniDMFMClassifier(self.model, freeze_backbone=True).train()
        self.assertFalse(frozen.backbone.training)
        self.assertFalse(frozen.backbone.backend.dropout.training)
        self.assertTrue(all(not p.requires_grad for p in frozen.backbone.parameters()))
        frozen(self.x).square().mean().backward()
        self.assertIsNotNone(frozen.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in frozen.backbone.parameters()))
        model = UniDMFMClassifier(UniDMFM(ProtocolBackend(), ['C3', 'Pz'])).train()
        model(self.x).square().mean().backward()
        self.assertGreater(model.backbone.backend.projection.weight.grad.abs().sum(), 0)

    def test_physical_metadata_and_incomplete_native_models_rejected(self):
        for names in [[], ['C3', 'c3'], [' C3'], ['C3', None], ['C3'] * 257]:
            with self.assertRaises(ValueError):
                UniDMFM(ProtocolBackend(), names)
        for rate in [100, 199, True]:
            with self.assertRaises(ValueError):
                UniDMFM(ProtocolBackend(), ['C3'], sampling_rate=rate)
        for x in [
            torch.zeros(0, 2, 200),
            torch.zeros(1, 1, 200),
            torch.zeros(1, 2, 199),
            torch.zeros(1, 2, 201),
            torch.zeros(1, 2, 20200),
            torch.ones(1, 2, 200, dtype=torch.int64),
            torch.full((1, 2, 200), float('inf')),
        ]:
            with self.assertRaises(ValueError):
                self.model(x)
        incomplete = ProtocolBackend()
        incomplete.contrastive_head = nn.Identity()
        with self.assertRaises(ValueError):
            UniDMFM(incomplete, ['C3'])
        with self.assertRaises(ValueError):
            self.model(self.x, return_tokens=1)


if __name__ == '__main__':
    unittest.main()
