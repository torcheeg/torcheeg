"""Complete original 386-state model and released pretraining-loop oracle."""

import copy
import importlib.util
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import UniDMFM, UniDMFMPretrainer


def load_example():
    path = (
        Path(__file__).resolve().parents[2]
        / 'examples/examples_eegmat_unidmfm_pretraining.py'
    )
    spec = importlib.util.spec_from_file_location('unidmfm_pretraining_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    os.environ.get('TORCHEEG_UNIDMFM_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_UNIDMFM_CHECKPOINT'),
    'Complete separately obtained original source and checkpoint required.',
)
class TestUniDMFMPretrainingNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.example = load_example()
        root, checkpoint = (
            os.environ['TORCHEEG_UNIDMFM_BACKEND_ROOT'],
            os.environ['TORCHEEG_UNIDMFM_CHECKPOINT'],
        )
        native, _ = cls.example.build_unidmfm_backend(
            root, checkpoint, ['Fp1', 'Fp2', 'F3']
        )
        cls.ops = cls.example.load_native_pretraining(root)
        backbone = UniDMFM.from_pretrained(checkpoint, native, ['Fp1', 'Fp2', 'F3'])
        cls.model = UniDMFMPretrainer(backbone, cls.ops.NTXentLoss(temperature=0.2))
        cls.reference = copy.deepcopy(native)
        cls.initial = torch.load(
            checkpoint, weights_only=True, map_location='cpu', mmap=True
        )['model']
        cls.values = torch.sin(torch.arange(2 * 3 * 20 * 200).float() / 31).reshape(
            2, 3, 4000
        )
        cls.params = SimpleNamespace(
            use_augmentations=True,
            time_shift_prob=0.5,
            max_time_shift_ratio=0.1,
            channel_dropout_ratio=0.2,
            noise_level=0.01,
        )

    @classmethod
    def tearDownClass(cls):
        for name in ['model', 'reference', 'initial', 'values']:
            delattr(cls, name)

    def setUp(self):
        self.model.backbone.backend.load_state_dict(self.initial, strict=True)
        self.reference.load_state_dict(self.initial, strict=True)
        self.model.zero_grad(set_to_none=True)
        self.reference.zero_grad(set_to_none=True)

    def test_complete_source_augmented_losses_input_all_parameter_gradients(self):
        self.assertEqual(len(self.model.backbone.backend.state_dict()), 386)
        self.assertEqual(len(self.model.state_dict()), 389)
        self.assertEqual(sum(v.numel() for v in self.initial.values()), 29421093)
        for training in [False, True]:
            self.setUp()
            self.model.train(training)
            self.reference.train(training)
            left = self.values.clone().requires_grad_()
            right = self.values.clone().requires_grad_()
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            before = self.example.capture_rng()
            first, second, mask = self.example.prepare_views(
                self.ops, left, self.params
            )
            actual = self.model(left, first, second, mask)
            self.example.restore_rng(before)
            clean = right.reshape(2, 3, 20, 200)
            original_first = self.ops.augmentations(clean, self.params)
            original_second = self.ops.augmentations(clean, self.params)
            original_mask = self.ops.generate_span_mask_vectorized(
                clean.shape[:3], 0.6, right.device
            )
            time, frequency, projection = self.reference(
                original_first * (1 - original_mask.unsqueeze(-1)), mode='full'
            )
            other = self.reference(original_second, mode='contrastive')
            time_loss = (
                F.mse_loss(time, clean, reduction='none') * original_mask.unsqueeze(-1)
            ).sum() / (original_mask.sum() * 200 + 1e-8)
            with torch.no_grad():
                stft = torch.stft(
                    clean.reshape(-1, 200),
                    n_fft=64,
                    hop_length=16,
                    window=torch.hann_window(64),
                    return_complex=True,
                )
                target = torch.log(stft.abs() + 1e-6).flatten(1)
            frequency_loss = (
                F.mse_loss(frequency.reshape(-1, 429), target, reduction='none')
                * original_mask.reshape(-1, 1)
            ).sum() / (original_mask.sum() * 429 + 1e-8)
            contrastive_loss = self.ops.NTXentLoss(temperature=0.2)(projection, other)
            expected_loss = (
                1.0 * time_loss + 1.0 * frequency_loss + 1.0 * contrastive_loss
            )
            for key, value in dict(
                mask=original_mask,
                time_prediction=time,
                frequency_prediction=frequency,
                first_projection=projection,
                second_projection=other,
                frequency_target=target,
                time_loss=time_loss,
                frequency_loss=frequency_loss,
                contrastive_loss=contrastive_loss,
                loss=expected_loss,
            ).items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)
            self.assertFalse(actual['frequency_target'].requires_grad)
            actual['loss'].backward()
            expected_loss.backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            reference = dict(self.reference.named_parameters())
            count = 0
            for name, parameter in self.model.backbone.backend.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(
                    parameter.grad, reference[name].grad, rtol=0, atol=0, msg=name
                )
                count += 1
            self.assertEqual(count, 242)
            for name, value in self.model.backbone.backend.state_dict().items():
                torch.testing.assert_close(
                    value, self.reference.state_dict()[name], rtol=0, atol=0, msg=name
                )

        self.model.eval()
        single = self.values[:1]
        first, second, _ = self.example.prepare_views(self.ops, single, self.params)
        with torch.no_grad():
            degenerate = self.model(single, first, second, torch.zeros(1, 3, 20))
        for name in ['time_loss', 'frequency_loss', 'contrastive_loss', 'loss']:
            self.assertEqual(degenerate[name].item(), 0, name)

    def test_actual_original_cpu_trainer_optimizer_clip_and_scheduler_step(self):
        if torch.cuda.is_available():
            self.skipTest(
                'Original trainer auto-selects CUDA; this reference verifies CPU float32.'
            )
        self.model.train()
        self.reference.train()
        with tempfile.TemporaryDirectory() as directory:
            params = SimpleNamespace(
                **vars(self.params),
                cuda=0,
                temperature=0.2,
                lr=1e-4,
                wdecay=0.05,
                epochs=1,
                warmup_epochs=0,
                model_dir=directory,
                mask_ratio=0.6,
                w_mae_t=1.0,
                w_mae_f=1.0,
                w_con=1.0,
            )
            trainer = self.ops.UniversalTrainer(
                params, [self.values.reshape(2, 3, 20, 200)], self.reference
            )
            self.assertFalse(trainer.scaler.is_enabled())
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-4, weight_decay=0.05
            )
            scheduler = self.ops.CosineLRScheduler(
                optimizer,
                t_initial=1,
                lr_min=1e-6,
                warmup_t=0,
                warmup_lr_init=1e-6,
                cycle_limit=1,
                t_in_epochs=False,
            )
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            before = self.example.capture_rng()
            optimizer.zero_grad(set_to_none=True)
            result = self.model(
                self.values,
                *self.example.prepare_views(self.ops, self.values, self.params),
            )
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step_update(0)
            self.example.restore_rng(before)
            trainer.train()  # Executes the complete untouched released training method.
            for name, value in self.model.backbone.backend.state_dict().items():
                torch.testing.assert_close(
                    value, self.reference.state_dict()[name], rtol=0, atol=0, msg=name
                )
            for key, value in optimizer.state_dict()['state'].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(
                        tensor,
                        trainer.optimizer.state_dict()['state'][key][name],
                        rtol=0,
                        atol=0,
                    )
            self.assertEqual(scheduler.state_dict(), trainer.scheduler.state_dict())
            payload = torch.load(
                Path(directory) / 'bfm_epoch_1.pth',
                weights_only=True,
                map_location='cpu',
                mmap=True,
            )
            self.assertEqual(len(payload['model']), 386)


if __name__ == '__main__':
    unittest.main()
