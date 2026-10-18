from argparse import Namespace
from contextlib import ExitStack
import copy
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.models import SleepDG
from torcheeg.trainers import SleepDGTrainer
from test.models.fixtures.sleepdg_reference.models.model import Model as AuthorModel
from test.models.fixtures.sleepdg_reference.alignment_reference import CORAL


def model():
    return SleepDG(sequence_length=3, hid_channels=8, num_heads=2,
                   mlp_channels=16, dropout=0.)


def options():
    return dict(logger=False, enable_checkpointing=False, enable_model_summary=False,
                enable_progress_bar=False, num_sanity_val_steps=0)


def loaders():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(5, 3, 2, 3000, generator=generator)
    y = torch.arange(15).reshape(5, 3) % 5
    domains = torch.tensor([0, 1, 2, 3, 0])
    return (DataLoader(TensorDataset(x, y, domains), batch_size=4),
            DataLoader(TensorDataset(x, y), batch_size=4))


class TinySequenceModel(nn.Module):
    num_classes = 5

    def __init__(self):
        super().__init__()
        self.extractor = nn.Linear(2, 4)
        self.classifier = nn.Linear(4, 5)
        self.scale = nn.Parameter(torch.tensor(.5))

    def forward_all(self, x):
        features = self.extractor(x.mean(-1))
        return self.classifier(features), x * self.scale, features

    def forward(self, x):
        return self.classifier(self.extractor(x.mean(-1)))


class OverflowRecorder(pl.Callback):
    def __init__(self):
        self.updates = 0
        self.history = []

    def on_train_start(self, trainer, module):
        trainer.optimizers[0].register_step_post_hook(self.after_step)

    def after_step(self, optimizer, args, kwargs):
        self.updates += 1

    def on_after_backward(self, trainer, module):
        if trainer.global_step == 0:
            next(module.model.parameters()).grad.fill_(float('inf'))

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.before = {name: p.detach().clone() for name, p in module.model.named_parameters()}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if batch_idx == 0:
            assert all(torch.equal(p, self.before[name]) for name, p in module.model.named_parameters())
        self.history.append((self.updates, trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                             trainer.precision_plugin.scaler.get_scale()))


class FiniteGradientRecorder(pl.Callback):
    def __init__(self):
        self.checks = 0

    def on_before_optimizer_step(self, trainer, module, optimizer):
        for name, parameter in module.model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        self.checks += 1


class TestSleepDGTrainer(unittest.TestCase):
    def test_native_complete_objective_and_all_gradients(self):
        torch.manual_seed(42)
        actual_model = SleepDG(dropout=0.).double().train()
        torch.manual_seed(42)
        reference = AuthorModel(Namespace(dropout=0., num_of_classes=5)).double().train()
        trainer = SleepDGTrainer(actual_model, verbose=False)
        x = torch.randn(2, 20, 2, 3000, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        labels = torch.arange(40).reshape(2, 20) % 5
        domains = torch.tensor([0, 1])
        actual = trainer.training_step((x, labels, domains), 0)
        logits, reconstruction, features = reference(other)
        expected = (F.cross_entropy(logits.transpose(1, 2), labels, label_smoothing=.1)
                    + F.mse_loss(reconstruction, other) + CORAL()(features, domains))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-7, atol=1e-9)
        for (name, parameter), (other_name, native) in zip(
                actual_model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, native.grad, rtol=1e-7, atol=1e-9, msg=name)
        for name, value in actual_model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)

    def test_real_lightning_fit_updates_encoder_decoder_and_classifier(self):
        train, val = loaders()
        trainer = SleepDGTrainer(model(), verbose=False)
        initial = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
        trainer.fit(train, val, max_epochs=1, **options())
        self.assertEqual(trainer.trainer.global_step, 2)
        self.assertEqual(trainer.trainer.gradient_clip_val, 1.)
        scheduler = trainer.trainer.lr_scheduler_configs[0].scheduler
        self.assertEqual(scheduler.T_max, 2)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertAlmostEqual(trainer.trainer.optimizers[0].param_groups[0]['lr'], 0.)
        for name in ('ae.encoder.epoch_encoder.encoder.0.weight', 'ae.encoder.fc_mu.weight',
                     'ae.decoder.upsample.0.weight', 'ae.decoder.upsample.20.weight',
                     'classifier.weight'):
            self.assertFalse(torch.equal(dict(trainer.model.named_parameters())[name], initial[name]), name)
        result = SleepDGTrainer(trainer.model, verbose=True, metrics=['accuracy', 'f1score'])
        measured = result.test(val, logger=False, enable_progress_bar=False,
                               enable_model_summary=False)[0]
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in measured.values()))
        self.assertIn('test_accuracy', measured)
        self.assertIn('test_f1score', measured)

    def test_evaluation_and_prediction_do_not_use_decoder_or_source_codes(self):
        train, val = loaders()
        trainer = SleepDGTrainer(model(), verbose=False).eval()
        x, y = next(iter(val))
        buffers = copy.deepcopy(trainer.model.state_dict())
        with patch.object(trainer.model, 'forward_all', side_effect=AssertionError('full path')):
            with patch.object(trainer.model.ae.decoder, 'forward', side_effect=AssertionError('decoder')):
                for step in (trainer.validation_step, trainer.test_step):
                    actual = step((x, y, torch.full((len(x),), 999)), 0)
                    expected = F.cross_entropy(trainer.model(x).flatten(0, 1), y.flatten(), label_smoothing=.1)
                    torch.testing.assert_close(actual, expected)
                for batch in (x, (x, y), (x, y, torch.full((len(x),), 999))):
                    torch.testing.assert_close(trainer.predict_step(batch, 0), trainer.model(x))
        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(tensor, buffers[name], rtol=0, atol=0)

    def test_eval_epoch_weighting_unequal_batches_and_logging_disabled(self):
        _, loader = loaders()
        trainer = SleepDGTrainer(model(), verbose=False, metrics=['accuracy', 'f1score']).eval()
        all_logits, all_labels = [], []
        for i, (x, y) in enumerate(loader):
            trainer.validation_step((x, y), i)
            trainer.test_step((x, y), i)
            all_logits.append(trainer.model(x).detach().flatten(0, 1))
            all_labels.append(y.flatten())
        expected = F.cross_entropy(torch.cat(all_logits), torch.cat(all_labels), label_smoothing=.1)
        torch.testing.assert_close(trainer.val_loss.compute(), expected)
        torch.testing.assert_close(trainer.test_loss.compute(), expected)
        expected_accuracy = (torch.cat(all_logits).argmax(1) == torch.cat(all_labels)).float().mean()
        torch.testing.assert_close(trainer.test_metrics.compute()['MulticlassAccuracy'], expected_accuracy)
        predictions, targets = torch.cat(all_logits).argmax(1), torch.cat(all_labels)
        expected_f1 = torch.stack([2 * ((predictions == c) & (targets == c)).sum()
                                  / ((predictions == c).sum() + (targets == c).sum())
                                  for c in range(5)]).mean()
        torch.testing.assert_close(trainer.test_metrics.compute()['MulticlassF1Score'], expected_f1)

    def test_weighted_components_and_float16_loss_statistics(self):
        train, _ = loaders()
        x, y, domains = next(iter(train))
        for reconstruction_weight, alignment_weight in ((0., 0.), (.3, 2.)):
            trainer = SleepDGTrainer(model(), verbose=False, weight_reconstruction=reconstruction_weight,
                                     weight_alignment=alignment_weight, label_smoothing=0.)
            trainer.eval()
            actual = trainer.training_step((x, y, domains), 0)
            logits, reconstruction, features = trainer.model.forward_all(x)
            expected = (F.cross_entropy(logits.flatten(0, 1), y.flatten())
                        + reconstruction_weight * F.mse_loss(reconstruction, x)
                        + alignment_weight * trainer.alignment_fn(features, domains))
            torch.testing.assert_close(actual, expected)
        trainer = SleepDGTrainer(model(), verbose=False)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = trainer.training_step((x, y, domains), 0)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in trainer.model.parameters()))

    def test_train_metric_weighting_and_epoch_reset_without_logging(self):
        trainer = SleepDGTrainer(model(), verbose=False).eval()
        train, _ = loaders()
        losses, weights = [], []
        for i, batch in enumerate(train):
            losses.append(trainer.training_step(batch, i).detach())
            weights.append(batch[1].numel())
        expected = sum(loss * count for loss, count in zip(losses, weights)) / sum(weights)
        torch.testing.assert_close(trainer.train_loss.compute(), expected)
        trainer.on_train_epoch_end()
        final = list(train)[-1]
        loss = trainer.training_step(final, 0)
        torch.testing.assert_close(trainer.train_loss.compute(), loss.detach())

    def test_invalid_model_outputs_cannot_broadcast_reconstruction_or_logits(self):
        trainer = SleepDGTrainer(model(), verbose=False).eval()
        train, _ = loaders()
        batch = next(iter(train))
        logits, reconstruction, features = trainer.model.forward_all(batch[0])
        cases = [(logits, reconstruction[:, :, :1], features),
                 (logits[:, :, :2], reconstruction, features),
                 (logits, reconstruction, features[:, :1]),
                 (logits.long(), reconstruction, features),
                 (logits, reconstruction), dict(logits=logits)]
        for outputs in cases:
            with patch.object(trainer.model, 'forward_all', return_value=outputs):
                with self.assertRaises(ValueError):
                    trainer.training_step(batch, 0)
        for value in (None, (), (batch[0],), torch.randn(2, 3)):
            with self.assertRaises(ValueError):
                trainer.predict_step(value, 0)

    def test_optimizer_configuration_explicit_horizon_and_disabled_scheduler(self):
        trainer = SleepDGTrainer(model(), verbose=False, scheduler_steps=4, optimizer_eps=1e-4)
        config = trainer.configure_optimizers()
        optimizer, scheduler = config['optimizer'], config['lr_scheduler']['scheduler']
        self.assertEqual(optimizer.defaults['eps'], 1e-4)
        self.assertEqual(optimizer.defaults['weight_decay'], 1e-4)
        self.assertEqual(config['lr_scheduler']['interval'], 'step')
        rates = []
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]['lr'])
        expected = [.001 * (1 + torch.cos(torch.tensor(torch.pi * step / 4))).item() / 2
                    for step in range(1, 5)]
        torch.testing.assert_close(torch.tensor(rates), torch.tensor(expected))
        without = SleepDGTrainer(model(), lr_scheduler=False, verbose=False).configure_optimizers()
        self.assertIsInstance(without, torch.optim.Adam)

    def test_invalid_batches_do_not_update_model_buffers(self):
        trainer = SleepDGTrainer(model(), verbose=False)
        train, _ = loaders()
        x, y, domains = next(iter(train))
        state = copy.deepcopy(trainer.model.state_dict())
        cases = [(x, y), (x, y[:, :2], domains), (x, y.float(), domains),
                 (x, y, domains[:, None]), (x, y, domains.float()),
                 (x, y, torch.full_like(domains, 4)), (x, y, -torch.ones_like(domains)),
                 (x, torch.full_like(y, 5), domains), (x, -torch.ones_like(y), domains),
                 (x[:, :, :, :0], y, domains)]
        for batch in cases:
            with self.assertRaises(ValueError):
                trainer.training_step(batch, 0)
        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)

    def test_constructor_validation_and_hyperparameters(self):
        for kwargs in [dict(lr=0), dict(lr=True), dict(weight_decay=-1),
                       dict(weight_reconstruction=float('nan')), dict(weight_alignment=-1),
                       dict(label_smoothing=1.1), dict(label_smoothing=True),
                       dict(optimizer_eps=0), dict(scheduler_steps=0),
                       dict(scheduler_steps=True), dict(lr_scheduler='yes'),
                       dict(num_classes=3), dict(num_domains=0)]:
            with self.assertRaises(ValueError):
                SleepDGTrainer(model(), **kwargs)
        with self.assertRaises(TypeError):
            SleepDGTrainer(nn.Identity())
        trainer = SleepDGTrainer(model(), scheduler_steps=10)
        self.assertNotIn('model', trainer.hparams)
        self.assertEqual(trainer.hparams['num_domains'], 4)
        self.assertEqual(trainer.hparams['scheduler_steps'], 10)

    def test_source_validation_checkpoint_is_explicitly_restored_for_testing(self):
        train, val = loaders()
        with tempfile.TemporaryDirectory() as root:
            callback = pl.callbacks.ModelCheckpoint(dirpath=root, monitor='val_accuracy',
                                                    mode='max', save_top_k=1)
            trainer = SleepDGTrainer(model(), verbose=True)
            fit_options = options()
            fit_options.pop('enable_checkpointing')
            trainer.fit(train, val, max_epochs=1, callbacks=[callback], **fit_options)
            self.assertTrue(callback.best_model_path)
            saved = torch.load(callback.best_model_path, weights_only=True)['state_dict']
            with torch.no_grad():
                trainer.model.classifier.weight.add_(3.)
            trainer.test(val, ckpt_path=callback.best_model_path, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, saved['model.' + name], rtol=0, atol=0)

    def test_lightning_checkpoint_restores_optimizer_scheduler_and_full_model(self):
        train, val = loaders()
        with tempfile.TemporaryDirectory() as root:
            first = SleepDGTrainer(model(), verbose=False, scheduler_steps=4)
            first.fit(train, val, max_epochs=2, max_steps=1, **options())
            state = copy.deepcopy(first.model.state_dict())
            path = Path(root, 'step.ckpt')
            first.trainer.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['global_step'], 1)
            self.assertTrue(payload['optimizer_states'][0]['state'])
            self.assertEqual(payload['lr_schedulers'][0]['last_epoch'], 1)
            restored = SleepDGTrainer(model(), verbose=False, scheduler_steps=4)
            restored.load_state_dict(payload['state_dict'], strict=True)
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            restored.fit(train, val, max_epochs=2, max_steps=2, ckpt_path=str(path), **options())
            restarted = restored.trainer
            self.assertEqual(restarted.global_step, 2)
            self.assertEqual(restarted.lr_scheduler_configs[0].scheduler.last_epoch, 2)
            self.assertEqual(restarted.lr_scheduler_configs[0].scheduler.T_max, 4)
            for value in restarted.optimizers[0].state_dict()['state'].values():
                self.assertEqual(value['step'].item(), 2)
            for name in ('ae.encoder.fc_mu.weight', 'classifier.weight'):
                self.assertFalse(torch.equal(restored.model.state_dict()[name], state[name]), name)
            self.assertTrue(any(not torch.equal(restored.model.state_dict()[name], value)
                                for name, value in state.items() if value.is_floating_point()))

    def test_amp_overflow_skips_parameter_and_learning_rate_updates(self):
        from pytorch_lightning.plugins.precision import MixedPrecision

        torch.manual_seed(42)
        x, y, domains = torch.randn(12, 3, 2, 4), torch.arange(36).reshape(12, 3) % 5, torch.arange(12) % 4
        train = DataLoader(TensorDataset(x, y, domains), batch_size=4)
        val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)
        record = OverflowRecorder()
        plugin = MixedPrecision('16-mixed', 'cpu', scaler=torch.amp.GradScaler('cpu', init_scale=32.))
        trainer = SleepDGTrainer(TinySequenceModel(), verbose=False, scheduler_steps=3)
        trainer.fit(train, val, max_epochs=1, plugins=[plugin], callbacks=[record], **options())
        self.assertEqual(record.history, [(0, 0, 16.), (1, 1, 16.), (2, 2, 16.)])
        self.assertEqual(trainer.trainer.global_step, 3)
        self.assertEqual(record.updates, 2)

    @unittest.skipUnless(os.getenv('TORCHEEG_TEST_COMPILE') == '1'
                         or os.getenv('GITHUB_ACTIONS') == 'true', 'Optional compiler check')
    def test_compiled_full_model_path_keeps_trainer_objective_and_gradients(self):
        if not hasattr(torch, 'compile'):
            self.skipTest('Requires PyTorch compile')
        from torch.nn.attention import SDPBackend, sdpa_kernel

        initial_mkldnn = torch.backends.mkldnn.enabled
        for dtype in (torch.float32, torch.float64):
            # CPU flash SDPA in PyTorch 2.6 has a compiled FP64 backward dtype
            # mismatch. Select the public math backend explicitly for FP64.
            with ExitStack() as context:
                # Use matching CPU reference convolution kernels for strict
                # gradient parity. The default optimized path is tested below.
                context.enter_context(torch.backends.mkldnn.flags(enabled=False))
                if dtype == torch.float64:
                    context.enter_context(sdpa_kernel(SDPBackend.MATH))
                torch.manual_seed(42)
                eager_model = model().to(dtype=dtype).train()
                compiled_model = copy.deepcopy(eager_model)
                compiled_model.forward_all = torch.compile(compiled_model.forward_all, fullgraph=True, dynamic=False)
                eager = SleepDGTrainer(eager_model, verbose=False)
                compiled = SleepDGTrainer(compiled_model, verbose=False)
                batch = list(next(iter(loaders()[0])))
                batch[0] = batch[0].to(dtype=dtype)
                expected, actual = eager.training_step(batch, 0), compiled.training_step(batch, 0)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                self.assertEqual(actual.dtype, dtype)
                expected.backward()
                actual.backward()
                pairs = list(zip(eager.model.named_parameters(), compiled.model.named_parameters()))
                if dtype == torch.float32:
                    # Full FP32 CE/MSE/alignment has large gradients; compare the
                    # raw model-relative error and the gradients actually used by
                    # the default norm-clipped optimizer. FP64 checks raw entries.
                    difference = sum((p.grad.double() - q.grad.double()).square().sum()
                                     for (_, p), (_, q) in pairs).sqrt()
                    reference_norm = sum(p.grad.double().square().sum() for (_, p), _ in pairs).sqrt()
                    self.assertLess((difference / reference_norm.clamp_min(1e-30)).item(), 1e-4)
                    for (name, parameter), (_, other) in pairs:
                        error = (parameter.grad.double() - other.grad.double()).norm()
                        scale = parameter.grad.double().norm().clamp_min(1e-3)
                        self.assertLess((error / scale).item(), 1e-4, name)
                    torch.nn.utils.clip_grad_norm_(eager.model.parameters(), 1.)
                    torch.nn.utils.clip_grad_norm_(compiled.model.parameters(), 1.)
                for (name, parameter), (other_name, other) in pairs:
                    self.assertEqual(name, other_name)
                    torch.testing.assert_close(parameter.grad, other.grad,
                                               rtol=1e-3 if dtype == torch.float32 else 1e-7,
                                               atol=1e-5 if dtype == torch.float32 else 1e-9, msg=name)
        self.assertEqual(torch.backends.mkldnn.enabled, initial_mkldnn)

    @unittest.skipUnless(os.getenv('TORCHEEG_TEST_COMPILE') == '1'
                         or os.getenv('GITHUB_ACTIONS') == 'true', 'Optional compiler check')
    def test_default_compiled_backend_completes_full_training_updates(self):
        if not hasattr(torch, 'compile'):
            self.skipTest('Requires PyTorch compile')
        torch.manual_seed(42)
        network = model()
        initial = {name: p.detach().clone() for name, p in network.named_parameters()}
        network.forward_all = torch.compile(network.forward_all, fullgraph=True, dynamic=False)
        trainer = SleepDGTrainer(network, verbose=False)
        train, val = loaders()
        recorder = FiniteGradientRecorder()
        trainer.fit(train, val, max_epochs=1, callbacks=[recorder], **options())
        self.assertEqual(recorder.checks, 2)
        self.assertEqual(trainer.trainer.global_step, 2)
        self.assertEqual(trainer.trainer.lr_scheduler_configs[0].scheduler.last_epoch, 2)
        for name, parameter in network.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all(), name)
        for state in trainer.trainer.optimizers[0].state_dict()['state'].values():
            self.assertEqual(state['step'].item(), 2)
        for name in ('ae.encoder.epoch_encoder.encoder.0.weight', 'ae.encoder.fc_mu.weight',
                     'ae.decoder.upsample.0.weight', 'ae.decoder.upsample.20.weight',
                     'classifier.weight'):
            self.assertFalse(torch.equal(dict(network.named_parameters())[name], initial[name]), name)


if __name__ == '__main__':
    unittest.main()
