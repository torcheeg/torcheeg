import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import BetaVAETrainer, CBetaVAETrainer


class DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.randint(0, 1)


class DummyEncoder(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_mu = nn.Linear(in_channels, out_channels)
        self.fc_var = nn.Linear(in_channels, out_channels)

    def forward(self, x, y=None):
        return self.fc_mu(x), self.fc_var(x)


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestBetaVAETrainer(unittest.TestCase):

    def test_beta_vae_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        encoder = DummyEncoder(120, 60)
        decoder = DummyModel(60, 120)

        trainer = BetaVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        trainer = BetaVAETrainer(encoder,
                                 decoder,
                                 metrics=["is", "fid"],
                                 metric_extractor=DummyModel(120, 10),
                                 metric_classifier=DummyModel(120, 1),
                                 metric_num_features=10)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

    def test_cbeta_vae_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        encoder = DummyEncoder(120, 60)
        decoder = DummyModel(60, 120)

        trainer = CBetaVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        trainer = CBetaVAETrainer(encoder,
                                  decoder,
                                  metrics=["is", "fid"],
                                  metric_extractor=DummyModel(120, 10),
                                  metric_classifier=DummyModel(120, 1),
                                  metric_num_features=10)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
