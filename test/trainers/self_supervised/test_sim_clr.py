import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import SimCLRTrainer


class _DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return {'eeg': (torch.randn(120), torch.randn(120)), 'labels': random.randint(0, 1)}


class _DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestSimCLRTrainer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_sim_clr_fit_runs_optimization_and_reports_finite_metrics(self):
        train_dataset = _DummyDataset()
        val_dataset = _DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        model = _DummyModel()
        trainer = SimCLRTrainer(model, extract_channels=10)
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all((torch.isfinite(torch.as_tensor(value)).all()
                        for value in trainer.trainer.logged_metrics.values())))


if __name__ == '__main__':
    unittest.main()
