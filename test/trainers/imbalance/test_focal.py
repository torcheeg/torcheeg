import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers.imbalance import FocalLossTrainer


class DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.randint(0, 1)


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestFocal(unittest.TestCase):

    def test_focal_loss_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = DummyModel()

        trainer = FocalLossTrainer(
            model,
            devices=1,
            class_frequency=train_loader,
            accelerator='cpu',
            num_classes=2,
            metrics=['f1score'])
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        trainer.test_step(next(iter(test_loader)), batch_idx=0)

        trainer = FocalLossTrainer(
            model,
            class_frequency=[10, 20],
            devices=1,
            accelerator='cpu',
            num_classes=2,
            metrics=['f1score'])
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        trainer.test_step(next(iter(test_loader)), batch_idx=0)


if __name__ == '__main__':
    unittest.main()
