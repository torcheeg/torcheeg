import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.trainers.domain_adaption.mmd_like import (_DualDataLoader,
                                                        _MMDLikeTrainer)


class TestDualDataLoader(unittest.TestCase):
    def test_target_restarts_and_longer_target_is_truncated(self):
        for target_size in (2, 7):
            with self.subTest(target_size=target_size):
                source = DataLoader(TensorDataset(
                    torch.arange(5)), batch_size=1)
                target = DataLoader(TensorDataset(
                    torch.arange(target_size)), batch_size=1)
                loader = _DualDataLoader(source, target)
                self.assertEqual(len(loader), 5)
                for _ in range(2):
                    pairs = list(loader)
                    self.assertEqual([int(a[0].item())
                                     for a, _ in pairs], list(range(5)))
                    self.assertEqual([int(b[0].item()) for _, b in pairs],
                                     [index % target_size for index in range(5)])


class TestMMDLikeTrainer(unittest.TestCase):
    def _make_trainer(self, **kwargs):
        return _MMDLikeTrainer(nn.Linear(3, 4), nn.Linear(4, 2),
                               num_classes=2, **kwargs)

    def test_weight_scheduler_and_disabled_schedule(self):
        for enabled in (False, True):
            with self.subTest(weight_scheduler=enabled):
                trainer = self._make_trainer(weight_domain=3., weight_scheduler=enabled,
                                             lr_scheduler_gamma=2.)
                trainer.num_batches = 4
                trainer.non_warmup_epochs = 2
                with patch.object(type(trainer), 'current_epoch', new_callable=property,
                                  fget=lambda self: 0):
                    trainer.on_train_batch_start(None, 0)
                    self.assertEqual(trainer.scheduled_weight_domain,
                                     0. if enabled else 1.)
                    trainer.on_train_batch_start(None, 2)
                self.assertGreater(trainer.scheduled_weight_domain, 0.)
                self.assertLess(trainer.lr_factor, 1.)

    def test_optimizer_has_distinct_classifier_rate_and_optional_scheduler(self):
        for gamma in (0., 1.):
            with self.subTest(gamma=gamma):
                trainer = self._make_trainer(
                    lr=0.002, lr_scheduler_gamma=gamma)
                configuration = trainer.configure_optimizers()
                optimizers = configuration[0] if gamma else configuration
                groups = optimizers[0].param_groups
                self.assertEqual([group['lr']
                                 for group in groups], [0.002, 0.02])
                self.assertEqual({id(p) for p in groups[0]['params']},
                                 {id(p) for p in trainer.extractor.parameters()})
                self.assertEqual({id(p) for p in groups[1]['params']},
                                 {id(p) for p in trainer.classifier.parameters()})
                if gamma:
                    self.assertEqual(len(configuration[1]), 1)

    def test_missing_domain_loss_hook_identifies_the_class(self):
        with self.assertRaisesRegex(NotImplementedError, '_MMDLikeTrainer'):
            self._make_trainer()._domain_loss_fn(torch.ones(2, 4), torch.ones(2, 4))


if __name__ == '__main__':
    unittest.main()
