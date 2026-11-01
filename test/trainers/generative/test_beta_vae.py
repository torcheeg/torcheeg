import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import BetaVAETrainer, CBetaVAETrainer


class _DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.randint(0, 1)


class _DummyEncoder(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_mu = nn.Linear(in_channels, out_channels)
        self.fc_var = nn.Linear(in_channels, out_channels)

    def forward(self, x, *args, **kwargs):
        return self.fc_mu(x), self.fc_var(x)


class _DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class TestBetaVAETrainer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_beta_vae_fit_runs_optimization_and_reports_finite_metrics(self):
        train_dataset = _DummyDataset()
        val_dataset = _DummyDataset()
        test_dataset = _DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        encoder = _DummyEncoder(120, 60)
        decoder = _DummyModel(60, 120)

        trainer = BetaVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test_step(next(iter(test_loader)), batch_idx=0)

        trainer = BetaVAETrainer(encoder,
                                 decoder,
                                 metrics=["is", "fid"],
                                 metric_extractor=_DummyModel(120, 10),
                                 metric_classifier=_DummyModel(120, 1),
                                 metric_num_features=10)
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test(test_loader)

    def test_cbeta_vae_fit_runs_optimization_and_reports_finite_metrics(self):
        train_dataset = _DummyDataset()
        val_dataset = _DummyDataset()
        test_dataset = _DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        encoder = _DummyEncoder(120, 60)
        decoder = _DummyModel(60, 120)

        trainer = CBetaVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test_step(next(iter(test_loader)), batch_idx=0)

        trainer = CBetaVAETrainer(encoder,
                                  decoder,
                                  metrics=["is", "fid"],
                                  metric_extractor=_DummyModel(120, 10),
                                  metric_classifier=_DummyModel(120, 1),
                                  metric_num_features=10)
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test_step(next(iter(test_loader)), batch_idx=0)


if __name__ == '__main__':
    unittest.main()
