import random
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.models import BCUNet, BUNet
from torcheeg.trainers import CDDPMTrainer, DDPMTrainer
from torcheeg.trainers.generative.ddpm import (LitEma, _betas_for_alpha_bar,
                                               _make_beta_schedule)


class _DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(4, 9, 9), random.randint(0, 1)


class _Extractor(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(
                                       128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        return x


class _Classifier(nn.Module):

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(
                                       128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

        self.lin1 = nn.Linear(9 * 9 * 64, 1024)
        self.lin2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class TestDDPMTrainer(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)
        """Set up common test data"""
        # Prepare minimal datasets and dataloaders for testing
        self.train_dataset = _DummyDataset(length=10)  # Reduced dataset size
        self.val_dataset = _DummyDataset(length=10)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=4)  # Smaller batch size
        self.val_loader = DataLoader(self.val_dataset, batch_size=4)

    def test_ddpm_trainer_basic(self):
        """Test basic functionality of DDPMTrainer"""
        model = BUNet(in_channels=4)
        trainer = DDPMTrainer(
            model,
            metric_extractor=_Extractor(),
            metric_classifier=_Classifier(),
            metric_num_features=9 * 9 * 64,
            metrics=['fid', 'is'],
            accelerator='cpu'
        )

        # Only test single training step
        loss = trainer.training_step(
            next(iter(self.train_loader)), batch_idx=0)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.device.type, 'cpu')
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

        # Only test single validation step
        trainer.validation_step(next(iter(self.val_loader)), batch_idx=0)

    def test_cddpm_trainer_basic(self):
        """Test basic functionality of CDDPMTrainer"""
        model = BCUNet(in_channels=4)
        trainer = CDDPMTrainer(
            model,
            metric_extractor=_Extractor(),
            metric_classifier=_Classifier(),
            metric_num_features=9 * 9 * 64,
            metrics=['fid', 'is'],
            accelerator='cpu'
        )

        # Only test single training step
        loss = trainer.training_step(
            next(iter(self.train_loader)), batch_idx=0)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.device.type, 'cpu')
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

        # Only test single validation step
        trainer.validation_step(next(iter(self.val_loader)), batch_idx=0)


class _LinearDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, y=None):
        return x * self.scale


class TestDiffusionMath(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(7)

    def _make_trainer(self, trainer_class=DDPMTrainer, **kwargs):
        return trainer_class(_LinearDenoiser(), timesteps=4, use_ema=False, **kwargs)

    def test_beta_schedules_follow_independent_formulas(self):
        start, end, steps = 0.001, 0.02, 4
        expected = {
            'linear': np.linspace(np.sqrt(start), np.sqrt(end), steps) ** 2,
            'sqrt_linear': np.linspace(start, end, steps),
            'sqrt': np.sqrt(np.linspace(start, end, steps)),
        }
        angles = (np.arange(steps + 1) / steps + 0.008) / 1.008 * np.pi / 2
        cumulative = np.cos(angles) ** 2
        expected['cosine'] = np.minimum(
            1 - cumulative[1:] / cumulative[:-1], 0.999)
        for schedule, reference in expected.items():
            with self.subTest(schedule=schedule):
                np.testing.assert_allclose(_make_beta_schedule(
                    schedule, steps, linear_start=start, linear_end=end), reference)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            _make_beta_schedule('unknown', steps)
        np.testing.assert_allclose(_betas_for_alpha_bar(4, lambda t: 1 - t / 2),
                                   [1 / 8, 1 / 7, 1 / 6, 1 / 5])

    def test_forward_noise_and_inverse_parameterizations_recover_original_signal(self):
        trainer = self._make_trainer()
        signal = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3) / 12
        noise = torch.full_like(signal, 0.25)
        t = torch.tensor([0, 3])
        alpha = torch.cumprod(1 - trainer.betas, dim=0)[t].reshape(2, 1, 1, 1)
        noisy = alpha.sqrt() * signal + (1 - alpha).sqrt() * noise
        torch.testing.assert_close(trainer.q_sample(signal, t, noise), noisy)
        torch.testing.assert_close(
            trainer.predict_start_from_noise(noisy, t, noise), signal)
        velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * signal
        torch.testing.assert_close(trainer.get_v(signal, noise, t), velocity)
        torch.testing.assert_close(
            trainer.predict_start_from_z_and_v(noisy, t, velocity), signal)
        torch.testing.assert_close(
            trainer.predict_eps_from_z_and_v(noisy, t, velocity), noise)
        mean, variance, log_variance = trainer.q_mean_variance(signal, t)
        torch.testing.assert_close(mean, alpha.sqrt() * signal)
        torch.testing.assert_close(variance, 1 - alpha)
        beta64 = np.linspace(np.sqrt(1e-4), np.sqrt(0.02), 4) ** 2
        log_reference = torch.tensor(
            np.log1p(-np.cumprod(1 - beta64)), dtype=torch.float32)
        torch.testing.assert_close(
            log_variance, log_reference[t].reshape(2, 1, 1, 1))
        self.assertEqual(trainer.q_sample(signal, t, None).shape, signal.shape)

    def test_losses_match_elementwise_reference_and_backpropagate(self):
        signal = torch.full((2, 1, 2, 3), 0.4)
        noise = torch.full_like(signal, 0.2)
        t = torch.tensor([1, 3])
        for trainer_class in (DDPMTrainer, CDDPMTrainer):
            for parameterization in ('eps', 'x0', 'v'):
                with self.subTest(trainer=trainer_class.__name__, parameterization=parameterization):
                    trainer = self._make_trainer(
                        trainer_class, parameterization=parameterization)
                    alpha = torch.cumprod(
                        1 - trainer.betas, dim=0)[t].reshape(2, 1, 1, 1)
                    noisy = alpha.sqrt() * signal + (1 - alpha).sqrt() * noise
                    targets = {'eps': noise, 'x0': signal,
                               'v': alpha.sqrt() * noise - (1 - alpha).sqrt() * signal}
                    reference = (
                        0.1 * noisy - targets[parameterization]).square().mean()
                    args = (signal, t, torch.tensor(
                        [0, 1])) if trainer_class is CDDPMTrainer else (signal, t)
                    loss, values = trainer.p_losses(*args, noise=noise)
                    torch.testing.assert_close(loss, reference)
                    torch.testing.assert_close(
                        values['simple_loss'], reference)
                    loss.backward()
                    self.assertTrue(torch.isfinite(trainer.model.scale.grad))
                    trainer.p_losses(*args)
        pred, target = torch.tensor([1., 3.]), torch.tensor([2., 1.])
        for loss_type in ('l1', 'l2'):
            trainer = self._make_trainer(loss_type=loss_type)
            expected = (pred - target).abs()
            if loss_type == 'l2':
                expected = expected.square()
            torch.testing.assert_close(trainer.get_loss(
                pred, target, mean=False), expected)
            torch.testing.assert_close(
                trainer.get_loss(pred, target), expected.mean())

    def test_sampling_has_expected_shape_and_no_noise_at_final_step(self):
        for trainer_class in (DDPMTrainer, CDDPMTrainer):
            for parameterization in ('eps', 'x0', 'v'):
                for clip in (False, True):
                    with self.subTest(trainer=trainer_class.__name__, parameterization=parameterization, clip=clip):
                        trainer = self._make_trainer(trainer_class, parameterization=parameterization,
                                                     clip_denoised=clip)
                        x = torch.full((2, 1, 2, 3), 0.3)
                        t = torch.zeros(2, dtype=torch.long)
                        labels = torch.tensor([0, 1])
                        args = (x, t, labels) if trainer_class is CDDPMTrainer else (
                            x, t)
                        mean, _, _ = trainer.p_mean_variance(
                            *args, clip_denoised=clip)
                        torch.testing.assert_close(
                            trainer.p_sample(*args, clip_denoised=clip), mean)
                        forward_args = (
                            x, labels) if trainer_class is CDDPMTrainer else (x,)
                        result = trainer(*forward_args)
                        self.assertEqual(result.shape, x.shape)
                        self.assertTrue(torch.isfinite(result).all())
                        trainer.p_sample(*args, repeat_noise=True)
                        batch = (x, labels)
                        self.assertEqual(trainer.predict_step(
                            batch, 0).shape, x.shape)

    def test_invalid_parameterization_and_loss_have_clear_errors(self):
        with self.assertRaises(ValueError):
            self._make_trainer(parameterization='unknown')
        trainer = self._make_trainer(loss_type='unknown')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            trainer.get_loss(torch.ones(2), torch.zeros(2))

    def test_learned_log_variance_is_in_the_optimizer(self):
        trainer = self._make_trainer(learn_logvar=True)
        ids = {id(p) for group in trainer.configure_optimizers(
        ).param_groups for p in group['params']}
        self.assertIn(id(trainer.logvar), ids)
        with trainer.ema_scope():
            self.assertIsInstance(trainer.model, _LinearDenoiser)

    def test_short_cpu_fit_test_and_prediction_cover_epoch_lifecycle(self):
        data = torch.arange(120, dtype=torch.float32).reshape(
            20, 1, 2, 3) / 120
        labels = torch.arange(20) % 2
        loader = DataLoader(torch.utils.data.TensorDataset(
            data, labels), batch_size=4)
        options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                       enable_model_summary=False)
        for trainer_class in (DDPMTrainer, CDDPMTrainer):
            with self.subTest(trainer=trainer_class.__name__):
                trainer = trainer_class(_LinearDenoiser(), timesteps=2,
                                        metric_extractor=nn.Flatten(), metric_num_features=6,
                                        metric_classifier=nn.Sequential(
                                            nn.Flatten(), nn.Linear(6, 2)),
                                        metrics=['fid', 'is'])
                trainer.fit(loader, loader, max_epochs=1, max_steps=1,
                            num_sanity_val_steps=0, **options)
                self.assertEqual(trainer.trainer.global_step, 1)
                metrics = trainer.test(loader, **options)[0]
                self.assertTrue(all(np.isfinite(value)
                                for value in metrics.values()))
                self.assertIn('test_fid', metrics)
                self.assertIn('test_is', metrics)


class TestLitEma(unittest.TestCase):
    def test_exponential_update_and_store_restore_match_scalar_reference(self):
        model = nn.Linear(1, 1)
        model.weight.data.fill_(1)
        model.bias.requires_grad_(False)
        ema = LitEma(model, decay=0.5, use_num_upates=False)
        model.weight.data.fill_(3)
        ema(model)
        ema.store(model.parameters())
        ema.copy_to(model)
        torch.testing.assert_close(model.weight, torch.tensor([[2.]]))
        ema.restore(model.parameters())
        torch.testing.assert_close(model.weight, torch.tensor([[3.]]))
        ema.reset_num_updates()
        self.assertEqual(ema.num_updates.item(), 0)
        ema(model)
        self.assertEqual(ema.num_updates.item(), 1)
        with self.assertRaises(ValueError):
            LitEma(model, decay=1.1)

    def test_ema_scope_restores_weights_even_after_failure(self):
        model = _LinearDenoiser()
        trainer = DDPMTrainer(model, timesteps=2, use_ema=True)
        model.scale.data.fill_(0.5)
        with self.assertRaisesRegex(RuntimeError, 'operation failed'):
            with trainer.ema_scope('test'):
                torch.testing.assert_close(model.scale, torch.tensor(0.1))
                raise RuntimeError('operation failed')
        torch.testing.assert_close(model.scale, torch.tensor(0.5))


if __name__ == '__main__':
    unittest.main()
