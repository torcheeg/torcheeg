"""Independent protocol fixtures for complete NeuroGPT interface behavior."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from torcheeg.models import NeuroGPT, NeuroGPTClassifier


CHANNELS = [
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


class ProtocolEncoder(nn.Module):
    """Small independent analytic fixture, not an author Conformer."""

    def __init__(self):
        super().__init__()
        self.n_chans, self.n_times, self.is_decoding_mode = 22, 500, False
        self.patch_embedding = nn.Module()
        self.patch_embedding.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)), nn.Conv2d(40, 40, (22, 1))
        )
        self.transformer = nn.ModuleList([nn.Identity() for _ in range(6)])
        for layer in self.transformer:
            layer.num_heads, layer.emb_size = 10, 40
        self.dropout = nn.Dropout(0.2)
        self.register_buffer('training_calls', torch.zeros((), dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.training_calls.add_(1)
        mean = x.mean((2, 3)).reshape(-1, 1, 1)
        scale = torch.linspace(0.5, 1.5, 1080, device=x.device, dtype=x.dtype).reshape(
            1, 27, 40
        )
        bias = sum(p.sum() for p in self.patch_embedding.parameters()) * 0.01
        return self.dropout(mean * scale + bias)


class ProtocolEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim, self.embed_dim, self.training_style = 1080, 1024, 'CSM'
        self.msk_embed = nn.Parameter(torch.full((1, 1, 1080), 0.1))
        self.cls_embed = nn.Parameter(torch.full((1, 1, 1080), 0.2))
        self.embed_model = nn.Module()
        self.embed_model.model = nn.Sequential(
            nn.Linear(1080, 1024), nn.LayerNorm(1024), nn.Dropout(0.1)
        )

    def duplicate_batch(self, batch):
        return batch

    def forward(self, batch):
        return self.embed_model.model(batch['inputs'])


class ProtocolDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_decoding_mode = False
        self.transformer = nn.Module()
        self.transformer.config = SimpleNamespace(
            n_embd=1024,
            n_layer=6,
            n_head=16,
            n_inner=4096,
            n_positions=512,
            activation_function='gelu_new',
            _attn_implementation='eager',
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
        )
        self.transformer.wte = nn.Embedding(1, 1024)
        self.transformer.wpe = nn.Embedding(512, 1024)
        self.pooler_layer = nn.Sequential(nn.Linear(1024, 1024), nn.Tanh())
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        x = batch['inputs_embeds'] * batch['attention_mask'].unsqueeze(-1)
        positions = self.transformer.wpe.weight[: x.shape[1]]
        return {'outputs': self.dropout(torch.tanh(x.cumsum(1) + positions))}


class ProtocolBackend(nn.Module):
    """Independent small interface fixture; never a pretrained substitute."""

    def __init__(self):
        super().__init__()
        self.encoder = ProtocolEncoder()
        self.embedder = ProtocolEmbedder()
        self.decoder = ProtocolDecoder()
        self.unembedder = nn.Module()
        self.unembedder.model = nn.Sequential(nn.Linear(1024, 1080))
        self.is_decoding_mode, self.ft_only_encoder = False, False

    def compute_loss(self, batch, return_outputs=False):
        x = batch['inputs']
        b, s = x.shape[:2]
        encoded = self.encoder(x).reshape(b, s, 1080).repeat_interleave(s - 1, dim=0)
        selected = torch.arange(1, s, device=x.device).repeat(b)
        positions = torch.arange(s, device=x.device).unsqueeze(0)
        target_mask = positions == selected.unsqueeze(1)
        valid = positions <= selected.unsqueeze(1)
        masked = torch.where(
            target_mask.unsqueeze(-1), self.embedder.msk_embed, encoded
        )
        masked = masked.masked_fill(~valid.unsqueeze(-1), 0)
        hidden = self.decoder(
            {
                'inputs_embeds': self.embedder({'inputs': masked}),
                'attention_mask': valid,
            }
        )['outputs']
        prediction = self.unembedder.model(hidden)
        loss = (prediction[target_mask] - encoded.detach()[target_mask]).square().mean()
        result = {'loss': loss, 'masking_loss': loss}
        return (result, {'outputs': prediction}) if return_outputs else result


class TestNeuroGPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.backend = ProtocolBackend()
        self.model = NeuroGPT(
            self.backend, CHANNELS, allow_montage_transfer=True
        ).eval()
        self.x = torch.randn(2, 3, 22, 500)

    def test_independent_encoder_order_last_feature_and_gradients(self):
        x = self.x.clone().requires_grad_()
        encoded = self.model.forward_encoder(x)
        bias = (
            sum(
                float(p.detach().sum())
                for p in self.backend.encoder.patch_embedding.parameters()
            )
            * 0.01
        )
        expected = (
            x.detach().numpy().mean((2, 3))[..., None, None]
            * np.linspace(0.5, 1.5, 1080, dtype=np.float32).reshape(27, 40)
            + bias
        )
        np.testing.assert_allclose(
            encoded.detach().numpy(), expected, rtol=1e-5, atol=1e-6
        )
        tokens = self.model(x, return_tokens=True)
        self.assertEqual(tokens.shape, (2, 3, 1024))
        torch.testing.assert_close(self.model(x), tokens[:, -1], rtol=0, atol=0)
        tokens.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum(), 0)
        self.assertGreater(
            self.backend.embedder.embed_model.model[0].weight.grad.abs().sum(), 0
        )
        self.assertEqual(self.model(x.transpose(0, 1)).shape, (3, 1024))

    def test_eval_raw_sample_and_causal_prefix_isolation(self):
        before = self.model(self.x, return_tokens=True)
        changed = self.x.clone()
        changed[1] += 2
        torch.testing.assert_close(
            self.model(changed, return_tokens=True)[0], before[0], rtol=0, atol=0
        )
        changed = self.x.clone()
        changed[:, -1] += 2
        torch.testing.assert_close(
            self.model(changed, return_tokens=True)[:, :-1],
            before[:, :-1],
            rtol=0,
            atol=0,
        )
        self.assertEqual(self.model(torch.zeros(1, 1, 22, 500)).shape, (1, 1024))

    def test_complete_reconstruction_loss_and_optimizer(self):
        self.model.train()
        x = self.x.clone().requires_grad_()
        result = self.model.forward_reconstruction(x)
        self.assertEqual(result['prediction'].shape, (4, 3, 1080))
        torch.testing.assert_close(
            result['loss'], result['reconstruction_loss'], rtol=0, atol=0
        )
        before = self.backend.unembedder.model[0].weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        result['loss'].backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(torch.equal(before, self.backend.unembedder.model[0].weight))
        self.assertGreater(self.backend.embedder.msk_embed.grad.abs().sum(), 0)
        self.assertIsNone(self.backend.embedder.cls_embed.grad)
        self.assertIsNone(self.backend.decoder.pooler_layer[0].weight.grad)

    def test_safe_complete_weights_and_failed_load_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            state = copy.deepcopy(self.backend.state_dict())
            torch.save(state, path)
            loaded = NeuroGPT.from_pretrained(
                path, ProtocolBackend(), CHANNELS, allow_montage_transfer=True
            ).eval()
            torch.testing.assert_close(
                loaded(self.x), self.model(self.x), rtol=0, atol=0
            )
            torch.save({'model': {'module.' + k: v for k, v in state.items()}}, path)
            NeuroGPT.from_pretrained(
                path,
                ProtocolBackend(),
                CHANNELS,
                prefix='module.',
                state_key='model',
                allow_montage_transfer=True,
            )
            variants = []
            missing = dict(state)
            missing.pop('unembedder.model.0.weight')
            variants.append(missing)
            extra = dict(state)
            extra['extra'] = torch.ones(1)
            variants.append(extra)
            for value in [
                torch.ones(1),
                state['unembedder.model.0.weight'].double(),
                torch.full_like(state['unembedder.model.0.weight'], float('nan')),
            ]:
                bad = dict(state)
                bad['unembedder.model.0.weight'] = value
                variants.append(bad)
            for bad in variants:
                torch.save(bad, path)
                backend = ProtocolBackend()
                before = copy.deepcopy(backend.state_dict())
                with self.assertRaises(ValueError):
                    NeuroGPT.from_pretrained(
                        path, backend, CHANNELS, allow_montage_transfer=True
                    )
                for key, value in before.items():
                    torch.testing.assert_close(
                        backend.state_dict()[key], value, rtol=0, atol=0
                    )
            from safetensors.torch import save_file

            safe = Path(directory) / 'weights.safetensors'
            save_file(state, str(safe))
            loaded = NeuroGPT.from_pretrained(
                safe, ProtocolBackend(), CHANNELS, allow_montage_transfer=True
            ).eval()
            torch.testing.assert_close(
                loaded(self.x), self.model(self.x), rtol=0, atol=0
            )

    def test_metadata_and_to_empty_strict_restoration(self):
        state = copy.deepcopy(self.model.state_dict())
        restored = (
            NeuroGPT(ProtocolBackend(), CHANNELS, allow_montage_transfer=True)
            .to_empty(device='cpu')
            .eval()
        )
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(self.x), self.model(self.x), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'channel order'):
            NeuroGPT(
                ProtocolBackend(), list(reversed(CHANNELS)), allow_montage_transfer=True
            ).load_state_dict(state, strict=True)
        changed = ProtocolBackend()
        changed.encoder.dropout.p = 0.3
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            NeuroGPT(changed, CHANNELS, allow_montage_transfer=True).load_state_dict(
                state, strict=True
            )

    def test_frozen_classifier_buffers_and_finetuning(self):
        frozen = NeuroGPTClassifier(self.model, freeze_backbone=True).train()
        self.assertFalse(frozen.backbone.training)
        before = copy.deepcopy(frozen.backbone.state_dict())
        frozen(self.x).square().mean().backward()
        self.assertIsNotNone(frozen.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in frozen.backbone.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(
                frozen.backbone.state_dict()[key], value, rtol=0, atol=0
            )
        fine = NeuroGPTClassifier(
            NeuroGPT(ProtocolBackend(), CHANNELS, allow_montage_transfer=True)
        ).train()
        fine(self.x).square().mean().backward()
        self.assertGreater(
            fine.backbone.backend.encoder.patch_embedding.shallownet[0]
            .weight.grad.abs()
            .sum(),
            0,
        )
        self.assertGreater(fine.backbone.backend.encoder.training_calls, 0)

    def test_physical_inputs_and_native_modes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'allow_montage_transfer'):
            NeuroGPT(ProtocolBackend(), CHANNELS)
        native = NeuroGPT(ProtocolBackend(), list(NeuroGPT.PRETRAINING_CHANNELS))
        self.assertTrue(native.is_pretraining_montage)
        self.assertFalse(native.allow_montage_transfer)
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            NeuroGPT(
                ProtocolBackend(),
                list(NeuroGPT.PRETRAINING_CHANNELS),
                allow_montage_transfer=True,
            ).load_state_dict(native.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            NeuroGPT(ProtocolBackend(), CHANNELS, allow_montage_transfer=1)
        wrong = list(NeuroGPT.PRETRAINING_CHANNELS)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        with self.assertRaises(ValueError):
            NeuroGPT(ProtocolBackend(), wrong)
        for channels in [[], CHANNELS[:-1], CHANNELS + [CHANNELS[0]], ['C3'] * 22]:
            with self.assertRaises(ValueError):
                NeuroGPT(ProtocolBackend(), channels)
        for rate in [100, 249, True]:
            with self.assertRaises(ValueError):
                NeuroGPT(ProtocolBackend(), CHANNELS, sampling_rate=rate)
        for x in [
            torch.zeros(0, 2, 22, 500),
            torch.zeros(1, 0, 22, 500),
            torch.zeros(1, 513, 22, 500),
            torch.zeros(1, 2, 21, 500),
            torch.zeros(1, 2, 22, 499),
            torch.ones(1, 2, 22, 500, dtype=torch.int64),
            torch.full((1, 2, 22, 500), float('nan')),
        ]:
            with self.assertRaises(ValueError):
                self.model(x)
        with self.assertRaises(ValueError):
            self.model.forward_reconstruction(self.x[:, :1])
        with self.assertRaises(ValueError):
            self.model(self.x, return_tokens=1)
        self.backend.is_decoding_mode = True
        with self.assertRaisesRegex(ValueError, 'pretraining configuration'):
            self.model(self.x)
        wrong_heads = ProtocolBackend()
        wrong_heads.encoder.transformer[0].num_heads = 5
        with self.assertRaisesRegex(ValueError, 'ten-head'):
            NeuroGPT(wrong_heads, CHANNELS, allow_montage_transfer=True)
        incomplete = ProtocolBackend()
        incomplete.unembedder = nn.Identity()
        with self.assertRaises(ValueError):
            NeuroGPT(incomplete, CHANNELS, allow_montage_transfer=True)


if __name__ == '__main__':
    unittest.main()
