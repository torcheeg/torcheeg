import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import GlowTrainer
from torcheeg.models import BGlow


class DummyDataset(Dataset):
    def __init__(self, length: int = 2):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(4, 32, 32), random.randint(0, 1)


class TestGlowTrainer(unittest.TestCase):
    def test_glow_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = BGlow(in_channels=4)

        trainer = GlowTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = GlowTrainer(model, device_ids=[0])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    def test_cglow_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = BGlow(in_channels=4, num_classes=2)

        trainer = GlowTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = GlowTrainer(model, device_ids=[0])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
