"""Protocol/numerical guards only; complete original models tested separately."""

import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from torcheeg.models import UniDMFM, UniDMFMPretrainer
from test.models.test_unidmfm_pretraining_native import load_example


class _ProtocolHeads(nn.Module):
    """Small test-only heads, not a Uni-DMFM implementation or native oracle."""

    def __init__(self):
        super().__init__()
        self.time = nn.Linear(200, 200)
        self.frequency = nn.Linear(200, 429)
        self.projection = nn.Linear(200, 256)

    def forward(self, x, mode):
        projection = self.projection(x.mean((1, 2)))
        if mode == 'contrastive':
            return projection
        context = x + x.mean((1, 2), keepdim=True)
        return self.time(context), self.frequency(context), projection


class _ProtocolBackbone(UniDMFM):
    def __init__(self):
        nn.Module.__init__(self)
        self.backend = _ProtocolHeads()
        self.num_electrodes = 2
        self._digest = hashlib.sha256(b'protocol-test-only').digest()
        self.register_buffer(
            'metadata_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )


class _ProtocolCriterion(nn.Module):
    """A test loss to check delegation, not a reimplementation of native NT-Xent."""

    temperature = 0.2

    def forward(self, left, right):
        return (left - right).square().mean()


class TestUniDMFMPretraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            self.model = UniDMFMPretrainer(
                _ProtocolBackbone(), _ProtocolCriterion(), 2.0, 0.5, 0.3
            )
            self.values = torch.randn(2, 2, 600)
        self.mask = torch.tensor(
            [
                [[True, False, True], [False, True, False]],
                [[False, True, False], [True, False, True]],
            ]
        )

    def test_independent_numpy_stft_masked_mse_weights_and_gradients(self):
        # Use float64 for the independent FFT formula; the full source oracle
        # separately enforces exact float32 predictions, gradients and updates.
        self.model.backbone.double()
        x = self.values.double().clone().requires_grad_()
        result = self.model(x, x + 0.1, x - 0.2, self.mask)
        patches = x.detach().numpy().reshape(-1, 200).astype(np.float64)
        padded = np.pad(patches, ((0, 0), (32, 32)), mode='reflect')
        frames = np.stack([padded[:, i : i + 64] for i in range(0, 201, 16)], axis=-1)
        frames *= self.model.stft_window.numpy()[None, :, None]
        target = np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6).reshape(-1, 429)
        np.testing.assert_allclose(
            result['frequency_target'].numpy(), target, rtol=1e-10, atol=1e-10
        )
        mask = self.mask.numpy().astype(float)
        time_error = result['time_prediction'].detach().numpy().astype(
            float
        ) - patches.reshape(2, 2, 3, 200)
        expected_time = (time_error**2 * mask[..., None]).sum() / (
            mask.sum() * 200 + 1e-8
        )
        freq_error = (
            result['frequency_prediction']
            .detach()
            .numpy()
            .reshape(-1, 429)
            .astype(float)
            - target
        )
        expected_frequency = (freq_error**2 * mask.reshape(-1, 1)).sum() / (
            mask.sum() * 429 + 1e-8
        )
        projected = (
            (result['first_projection'] - result['second_projection']).detach().double()
        )
        expected_contrastive = projected.square().mean().item()
        self.assertAlmostEqual(result['time_loss'].item(), expected_time, places=5)
        self.assertAlmostEqual(
            result['frequency_loss'].item(), expected_frequency, places=5
        )
        self.assertAlmostEqual(
            result['loss'].item(),
            2 * expected_time + 0.5 * expected_frequency + 0.3 * expected_contrastive,
            places=5,
        )
        self.assertFalse(result['frequency_target'].requires_grad)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        # The test criterion subtracts two projections, so their shared bias
        # cancels exactly. Nonzero head WEIGHT gradients establish each path.
        for head in [
            self.model.backbone.backend.time,
            self.model.backbone.backend.frequency,
            self.model.backbone.backend.projection,
        ]:
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_empty_and_full_masks_preserve_native_denominators(self):
        empty = self.model(
            self.values,
            self.values + 0.1,
            self.values - 0.2,
            torch.zeros_like(self.mask),
        )
        self.assertEqual(empty['time_loss'].item(), 0)
        self.assertEqual(empty['frequency_loss'].item(), 0)
        torch.testing.assert_close(
            empty['loss'], 0.3 * empty['contrastive_loss'], rtol=0, atol=0
        )
        full = self.model(
            self.values,
            self.values + 0.1,
            self.values - 0.2,
            torch.ones_like(self.mask),
        )
        self.assertGreater(full['time_loss'].item(), 0)
        self.assertGreater(full['frequency_loss'].item(), 0)

    def test_view_mask_and_constructor_contracts(self):
        for bad in [
            torch.zeros(2, 2, 2),
            self.mask.long(),
            self.mask.float() + 0.1,
            torch.full(self.mask.shape, float('nan')),
        ]:
            with self.assertRaises(ValueError):
                self.model(self.values, self.values, self.values, bad)
        for bad in [
            self.values[:, :1],
            self.values.double(),
            torch.full_like(self.values, float('nan')),
        ]:
            with self.assertRaises(ValueError):
                self.model(self.values, bad, self.values, self.mask)
        with self.assertRaises(ValueError):
            UniDMFMPretrainer(nn.Identity(), _ProtocolCriterion())
        for value in [-1.0, float('nan'), True]:
            with self.assertRaises(ValueError):
                UniDMFMPretrainer(
                    _ProtocolBackbone(), _ProtocolCriterion(), time_weight=value
                )
        for value in [0.0, float('nan'), True]:
            criterion = _ProtocolCriterion()
            criterion.temperature = value
            with self.assertRaises(ValueError):
                UniDMFMPretrainer(_ProtocolBackbone(), criterion)
        criterion = _ProtocolCriterion()
        criterion.extra = nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            UniDMFMPretrainer(_ProtocolBackbone(), criterion)
        with self.assertRaises(ValueError):
            UniDMFMPretrainer(
                _ProtocolBackbone(), _ProtocolCriterion(), stft_window=torch.ones(63)
            )
        self.model.contrastive_loss.temperature = 0.3
        with self.assertRaisesRegex(ValueError, 'temperature changed'):
            self.model(self.values, self.values, self.values, self.mask)

    def test_window_snapshot_strict_policy_and_meta_restore(self):
        window = torch.linspace(0, 1, 64)
        model = UniDMFMPretrainer(
            _ProtocolBackbone(), _ProtocolCriterion(), stft_window=window
        )
        window.zero_()
        self.assertGreater(model.stft_window.sum().item(), 0)
        state = copy.deepcopy(self.model.state_dict())
        restored = UniDMFMPretrainer(
            _ProtocolBackbone(), _ProtocolCriterion(), 2.0, 0.5, 0.3
        )
        restored.to('meta').to_empty(device='cpu')
        restored.load_state_dict(state, strict=True)
        left = self.model(self.values, self.values + 0.1, self.values - 0.2, self.mask)
        right = restored(self.values, self.values + 0.1, self.values - 0.2, self.mask)
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        state['objective_fingerprint'][0] ^= 1
        with self.assertRaisesRegex(RuntimeError, 'objective'):
            restored.load_state_dict(state, strict=True)

    def test_example_three_rng_safe_checkpoint_roundtrip(self):
        example = load_example()
        before = example.capture_rng()
        try:
            torch.manual_seed(17)
            random.seed(17)
            np.random.seed(17)
            state = example.capture_rng()
            expected = (torch.rand(4), random.random(), np.random.rand(4))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'rng.pt'
                example.save_checkpoint(path, dict(rng=state))
                loaded = torch.load(path, weights_only=True, map_location='cpu')
                example.restore_rng(loaded['rng'])
                torch.testing.assert_close(torch.rand(4), expected[0], rtol=0, atol=0)
                self.assertEqual(random.random(), expected[1])
                np.testing.assert_array_equal(np.random.rand(4), expected[2])
                self.assertEqual(list(Path(directory).iterdir()), [path])
        finally:
            example.restore_rng(before)

    def test_example_source_normalization_and_outlier_clipping(self):
        example = load_example()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            values = torch.randn(19, 4000).numpy().astype(np.float64)
        expected = (values - values.mean(1, keepdims=True)) / (
            values.std(1, ddof=1, keepdims=True) + 1e-6
        )
        np.testing.assert_allclose(
            example.prepare_window(values).numpy(), expected, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            example.prepare_window(np.ones((19, 4000)) * 7),
            torch.zeros(19, 4000),
            rtol=0,
            atol=0,
        )
        values[:, 0] = 1e10
        self.assertEqual(example.prepare_window(values).max().item(), 10)
        with self.assertRaises(ValueError):
            example.prepare_window(np.zeros((19, 800)))

    def test_example_cli_rejects_invalid_counts_and_output_overwrite(self):
        example = load_example()
        base = [
            'example',
            '--root-path',
            'missing',
            '--backend-root',
            'missing',
            '--checkpoint',
            'missing',
            '--output-path',
            'new.pt',
        ]
        for args in [
            ['--steps', '0'],
            ['--batch-size', '1'],
            ['--validation-windows', '1'],
        ]:
            with patch('sys.argv', base + args):
                with self.assertRaises(SystemExit):
                    example.main()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.pt'
            path.write_text('preserve')
            with patch('sys.argv', base[:-1] + [str(path)]):
                with self.assertRaises(SystemExit):
                    example.main()
            self.assertEqual(path.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
