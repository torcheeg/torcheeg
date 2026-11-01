import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.trainers.domain_adaption.dann_like import (GradientReverse,
                                                         _DANNLikeTrainer,
                                                         _DualDataLoader)


class TestDualDataLoader(unittest.TestCase):
    def test_shorter_target_loader_restarts_for_each_source_batch(self):
        source = DataLoader(TensorDataset(torch.arange(5)), batch_size=1)
        target = DataLoader(TensorDataset(
            torch.tensor([10, 11])), batch_size=1)
        loader = _DualDataLoader(source, target)
        self.assertEqual(len(loader), 5)
        for _ in range(2):
            pairs = list(loader)
            self.assertEqual([int(a[0].item())
                             for a, _ in pairs], list(range(5)))
            self.assertEqual([int(b[0].item()) for _, b in pairs],
                             [10, 11, 10, 11, 10])


class TestGradientReverse(unittest.TestCase):
    def test_forward_is_identity_and_gradient_is_scaled_and_reversed(self):
        for alpha in (0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                value = torch.tensor([1., 2.], requires_grad=True)
                weight = torch.tensor([2., -3.])
                result = GradientReverse.apply(value, alpha)
                torch.testing.assert_close(result, value)
                (result * weight).sum().backward()
                torch.testing.assert_close(value.grad, -alpha * weight)


class TestDANNLikeTrainer(unittest.TestCase):
    def _make_trainer(self, **kwargs):
        return _DANNLikeTrainer(nn.Linear(3, 4), nn.Linear(4, 2),
                                nn.Linear(4, 2), num_classes=2, **kwargs)

    def test_schedulers_respect_warmup_and_progress(self):
        trainer = self._make_trainer(warmup_epochs=1, lr_scheduler_gamma=2.)
        trainer.num_batches = 4
        trainer.non_warmup_epochs = 2
        with patch.object(type(trainer), 'current_epoch', new_callable=property,
                          fget=lambda self: 0):
            trainer.on_train_batch_start(None, 2)
        self.assertEqual(trainer.scheduled_alpha, 1.)
        with patch.object(type(trainer), 'current_epoch', new_callable=property,
                          fget=lambda self: 1):
            trainer.on_train_batch_start(None, 0)
            self.assertEqual(trainer.scheduled_alpha, 0.)
            trainer.on_train_batch_start(None, 2)
        self.assertGreater(trainer.scheduled_alpha, 0.)
        self.assertLess(trainer.scheduled_alpha, 1.)
        self.assertLess(trainer.lr_factor, 1.)

    def test_optimizer_contains_all_network_parameters_and_optional_scheduler(self):
        for gamma in (0., 1.):
            with self.subTest(gamma=gamma):
                trainer = self._make_trainer(lr_scheduler_gamma=gamma)
                configuration = trainer.configure_optimizers()
                optimizers = configuration[0] if gamma else configuration
                ids = {id(parameter) for group in optimizers[0].param_groups
                       for parameter in group['params']}
                expected = {id(parameter) for module in (
                    trainer.extractor, trainer.classifier, trainer.domain_classifier)
                    for parameter in module.parameters()}
                self.assertEqual(ids, expected)
                if gamma:
                    self.assertEqual(len(configuration[1]), 1)

    def test_missing_domain_loss_hook_identifies_the_class(self):
        with self.assertRaisesRegex(NotImplementedError, '_DANNLikeTrainer'):
            self._make_trainer()._domain_loss_fn(torch.ones(2, 4), torch.ones(2, 4))


if __name__ == '__main__':
    unittest.main()
