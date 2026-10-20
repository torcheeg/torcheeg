"""Independent protocol fixtures; complete author models have a separate oracle."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from torcheeg.models import SleepGPT, SleepGPTClassifier


class ProtocolTransformer(nn.Module):
    """Small independent interface fixture, not a SleepGPT architecture replica."""

    def __init__(self):
        super().__init__()
        self.embed_dim, self.num_heads, self.tfffn_start_layer_index = 768, 16, 10
        (
            self.max_channels,
            self.num_patches,
            self.patch_size,
            self.fs,
            self.epoch_duration,
        ) = 8, 15, 200, 100, 30
        self.actual_channels = None
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.projection = nn.Linear(200, 768)
        self.mask_token = nn.Parameter(torch.linspace(-0.1, 0.1, 768))


class ProtocolBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = ProtocolTransformer()
        self.hparams = SimpleNamespace(
            config=dict(
                use_cb=False,
                time_only=False,
                fft_only=False,
                mask_ratio=0.75,
                drop_path_rate=0.1,
                loss_function='l2',
            )
        )
        self.pooler = nn.Identity()
        self.Masked_docoder = nn.Linear(768, 200)
        self.Masked_docoder_fft = nn.Linear(768, 200)

    def get_attention_mask(self, time, frequency):
        cls = torch.ones(len(time), 1, device=time.device)
        return [
            cls,
            time.repeat_interleave(15, 1),
            cls,
            frequency.repeat_interleave(15, 1),
        ]

    def infer(self, batch, time_mask=False, stage='test'):
        raw, frequency = batch['epochs']
        b = len(raw)
        time = self.transformer.projection(raw.reshape(b, 120, 200))
        freq = self.transformer.projection(frequency.reshape(b, 120, 200))
        valid = batch['mask'][1]
        if time_mask:
            tmask = batch['random_mask'][0]
            fmask = (torch.rand(b, 120, device=raw.device) < 0.75).to(raw.dtype) * valid
            time = torch.where(
                tmask.bool().unsqueeze(-1), self.transformer.mask_token, time
            )
            freq = torch.where(
                fmask.bool().unsqueeze(-1), self.transformer.mask_token, freq
            )

        def tokens(values):
            cls = (values * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            return torch.cat([cls.unsqueeze(1), values], 1)

        output = dict(time_feats=tokens(time), fft_feats=tokens(freq))
        if time_mask:
            output.update(
                cls_feats=self.Masked_docoder(time),
                cls_feats_fft=self.Masked_docoder_fft(freq),
                time_mask_patch=tmask,
                fft_mask_patch=fmask,
            )
        return output

    def forward_masked_loss(self, prediction, target, mask):
        error = (prediction - target.reshape(len(target), 120, 200)).square().mean(-1)
        return (error * mask).sum() / mask.sum()

    def forward_masked_loss_2D(self, prediction, target, mask):
        return self.forward_masked_loss(prediction, target, mask)


class TestSleepGPT(unittest.TestCase):
    def make(self, names=('C3', 'Pz')):
        return SleepGPT(ProtocolBackend(), list(names))

    def test_independent_spectral_definition_and_detachment(self):
        rng = np.random.RandomState(5)
        values = rng.normal(size=(1, 2, 3000))
        # Independent NumPy DFT, periodic window and explicit reflect frames.
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(200) / 200)
        window = np.pad(window, (28, 28))
        padded = np.pad(values, ((0, 0), (0, 0), (128, 128)), mode='reflect')
        frames = np.stack(
            [padded[..., i * 100 : i * 100 + 256] * window for i in range(31)], axis=-2
        )
        spectrum = np.fft.rfft(frames, n=256, axis=-1)[..., 1:, :100]
        log = 10 * np.log10(np.sqrt(spectrum.real**2 + spectrum.imag**2) + 1e-8)
        expected = (log - log.mean(-1, keepdims=True)) / log.std(
            -1, keepdims=True, ddof=1
        )
        x = torch.tensor(values, requires_grad=True)
        actual = SleepGPT.spectrogram(x)
        # Native Hann is computed in default float32; NumPy's independent
        # analytic Hann uses float64. Complete native oracle also checks exact
        # per-entry agreement against the source's actual float32 window.
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(
            SleepGPT.spectrogram(torch.zeros(1, 2, 3000)),
            torch.zeros(1, 2, 30, 100),
            rtol=0,
            atol=0,
        )

    def test_features_mapping_tokens_and_input_gradients(self):
        model = self.make(('Pz', 'C3')).double()
        x = torch.randn(2, 2, 3000, dtype=torch.float64, requires_grad=True)
        actual = model(x)
        mapped = torch.zeros(2, 8, 3000, dtype=x.dtype).index_copy(
            1, torch.tensor([7, 0]), x
        )
        present = torch.zeros(2, 8, dtype=x.dtype)
        present[:, [7, 0]] = 1
        batch = dict(
            epochs=(mapped, SleepGPT.spectrogram(mapped)),
            mask=model.backend.get_attention_mask(present, present),
        )
        expected = model.backend.infer(batch)
        target = torch.cat(
            [expected['time_feats'][:, 0], expected['fft_feats'][:, 0]], -1
        )
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        a = torch.autograd.grad(
            actual.square().sum(),
            (x, model.backend.transformer.projection.weight),
            retain_graph=True,
        )
        b = torch.autograd.grad(
            target.square().sum(), (x, model.backend.transformer.projection.weight)
        )
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        tokens = model.forward_tokens(x)
        self.assertEqual(tokens['time_tokens'].shape, (2, 8, 15, 768))
        self.assertEqual(tokens['frequency_tokens'].shape, (2, 8, 15, 768))
        self.assertTrue(tokens['padding_mask'][:, [1, 2, 3, 4, 5, 6]].all())
        self.assertFalse(tokens['padding_mask'][:, [0, 7]].any())

    def test_absent_sensor_values_cannot_change_output_or_gradients(self):
        model = self.make()
        x = torch.randn(2, 2, 3000, requires_grad=True)
        pad = torch.tensor([[False, True], [False, False]])
        before = model(x, pad)
        changed = x.detach().clone()
        changed[0, 1] = float('nan')
        torch.testing.assert_close(model(changed, pad), before, rtol=0, atol=0)
        before.sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[0, 1]), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_both_reconstruction_heads_loss_and_optimizer(self):
        model = self.make()
        x = torch.randn(2, 2, 3000, requires_grad=True)
        mask = torch.zeros(2, 2, 15, dtype=torch.bool)
        mask[:, :, ::3] = True
        result = model.forward_masked(x, mask)
        self.assertEqual(result['time_prediction'].shape, (2, 120, 200))
        self.assertEqual(result['frequency_prediction'].shape, (2, 120, 200))
        self.assertEqual(result['time_mask'].shape, (2, 8, 15))
        self.assertEqual(
            torch.count_nonzero(result['time_mask'][:, [1, 2, 3, 4, 5, 6]]), 0
        )
        torch.testing.assert_close(
            result['loss'],
            result['time_loss'] + result['frequency_loss'],
            rtol=0,
            atol=0,
        )
        original = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        result['loss'].backward()
        optimizer.step()
        for name in [
            'backend.Masked_docoder.weight',
            'backend.Masked_docoder_fft.weight',
        ]:
            current = dict(model.named_parameters())[name]
            self.assertIsNotNone(current.grad)
            self.assertTrue(torch.isfinite(current.grad).all())
            self.assertFalse(torch.equal(current, original[name]))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_complete_checkpoint_and_configuration_restore(self):
        model = self.make().eval()
        x = torch.randn(2, 2, 3000)
        state = copy.deepcopy(model.state_dict())
        restored = self.make().eval()
        restored.to_empty(device='cpu')
        restored.load_state_dict(state)
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            self.make(('Pz', 'C3')).load_state_dict(state)
        other = self.make()
        other.backend.hparams.config['loss_function'] = 'l1'
        other = SleepGPT(other.backend, ['C3', 'Pz'])
        with self.assertRaises(RuntimeError):
            other.load_state_dict(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            native = model.backend.state_dict()
            torch.save({'state_dict': native}, path)
            SleepGPT.from_pretrained(path, ProtocolBackend(), ['C3', 'Pz'])
            for bad in [
                dict(list(native.items())[1:]),
                dict(native, extra=torch.zeros(1)),
                dict(
                    native,
                    **{
                        'Masked_docoder.weight': native[
                            'Masked_docoder.weight'
                        ].double()
                    },
                ),
                dict(
                    native,
                    **{
                        'Masked_docoder.weight': torch.full_like(
                            native['Masked_docoder.weight'], float('nan')
                        )
                    },
                ),
            ]:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    SleepGPT.from_pretrained(
                        path, ProtocolBackend(), ['C3', 'Pz'], state_key=None
                    )
            torch.save({'module.' + k: v for k, v in native.items()}, path)
            SleepGPT.from_pretrained(
                path, ProtocolBackend(), ['C3', 'Pz'], state_key=None, prefix='module.'
            )
            from safetensors.torch import save_file

            path = Path(directory) / 'weights.safetensors'
            save_file(native, str(path))
            SleepGPT.from_pretrained(
                path, ProtocolBackend(), ['C3', 'Pz'], state_key=None
            )

    def test_classifier_freezing_and_fine_tuning(self):
        model = SleepGPTClassifier(self.make(), 3, freeze_backbone=True).train()
        self.assertFalse(model.backbone.training)
        loss = nn.functional.cross_entropy(
            model(torch.randn(2, 2, 3000)), torch.tensor([0, 2])
        )
        loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model = SleepGPTClassifier(self.make(), freeze_backbone=False)
        model(torch.randn(2, 2, 3000)).sum().backward()
        self.assertIsNotNone(model.backbone.backend.transformer.projection.weight.grad)

    def test_invalid_physical_inputs_backend_and_masks(self):
        model = self.make()
        for bad in [
            torch.randn(1, 2, 2999),
            torch.randn(0, 2, 3000),
            torch.randn(1, 3, 3000),
            torch.ones(1, 2, 3000, dtype=torch.long),
        ]:
            with self.assertRaises(ValueError):
                model(bad)
        with self.assertRaises(ValueError):
            self.make(('EEG Fpz-Cz',))
        with self.assertRaises(ValueError):
            self.make(('C3', 'c3'))
        with self.assertRaises(ValueError):
            SleepGPT(ProtocolBackend(), ['C3'], sampling_rate=200)
        backend = ProtocolBackend()
        backend.hparams.config['use_cb'] = True
        with self.assertRaises(ValueError):
            SleepGPT(backend, ['C3'])
        x = torch.randn(1, 2, 3000)
        with self.assertRaises(ValueError):
            model(x, torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model.forward_masked(x, torch.zeros(1, 2, 15, dtype=torch.bool))
        mask = torch.ones(1, 2, 15, dtype=torch.bool)
        with self.assertRaises(ValueError):
            model.forward_masked(x, mask, torch.tensor([[False, True]]))
        with self.assertRaises(ValueError):
            model.forward_masked(x, mask.long())


if __name__ == '__main__':
    unittest.main()
