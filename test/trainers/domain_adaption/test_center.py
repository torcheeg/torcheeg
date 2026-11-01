import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import CenterLossTrainer


class _DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.randint(0, 1)


class _DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestCenter(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_center_loss_fit_runs_optimization_and_reports_finite_metrics(self):

        train_dataset = _DummyDataset(length=10)
        val_dataset = _DummyDataset(length=10)
        test_dataset = _DummyDataset(length=10)

        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        test_loader = DataLoader(test_dataset, batch_size=5)

        extractor = _DummyModel(120, 64)
        classifier = _DummyModel(64, 2)

        trainer = CenterLossTrainer(extractor=extractor,
                                    classifier=classifier,
                                    feature_dim=64,
                                    num_classes=2)
        trainer.fit(train_loader, val_loader, max_epochs=2)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
