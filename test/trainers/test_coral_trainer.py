import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import CoralTrainer


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
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestCoralTrainer(unittest.TestCase):
    def test_base_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)

        trainer = CoralTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        score = trainer.score(test_loader)

        print(score)
        self.assertTrue(score <= 1)

        trainer = CoralTrainer(extractor,
                               classifier,
                               device=torch.device('cuda'))
        trainer.fit(source_loader, target_loader, val_loader)
        score = trainer.score(test_loader)
        print(score)

        self.assertTrue(score <= 1)


if __name__ == '__main__':
    unittest.main()
