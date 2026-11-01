import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import CenterLossTrainer


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


class TestCenter(unittest.TestCase):

    def test_center_loss_trainer(self):

        train_dataset = DummyDataset(length=10)
        val_dataset = DummyDataset(length=10)
        test_dataset = DummyDataset(length=10)

        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        test_loader = DataLoader(test_dataset, batch_size=5)

        extractor = DummyModel(120, 64)
        classifier = DummyModel(64, 2)

        trainer = CenterLossTrainer(extractor=extractor,
                                    classifier=classifier,
                                    feature_dim=64,
                                    num_classes=2)
        trainer.fit(train_loader, val_loader, max_epochs=2)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
