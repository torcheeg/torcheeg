import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import VAETrainer


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

    def forward(self, x):
        return self.fc_mu(x), self.fc_var(x)


class DummyModel(nn.Module):
    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestVAETrainer(unittest.TestCase):
    def test_base_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        encoder = DummyEncoder(120, 60)
        decoder = DummyModel(60, 1)

        trainer = VAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = VAETrainer(encoder, decoder, device_ids=[0])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
