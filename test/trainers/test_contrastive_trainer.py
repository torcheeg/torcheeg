import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import ContrastiveTrainer


class DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return (torch.randn(120), torch.randn(120)), random.randint(0, 1)


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestClassificationTrainer(unittest.TestCase):

    def test_classification_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)

        model = DummyModel()

        trainer = ContrastiveTrainer(model)
        trainer.fit(train_loader, val_loader, max_epochs=1)

        trainer = ContrastiveTrainer(model,
                                     devices=1,
                                     accelerator='gpu')
        trainer.fit(train_loader, val_loader,
                                     max_epochs=1)


if __name__ == '__main__':
    unittest.main()
