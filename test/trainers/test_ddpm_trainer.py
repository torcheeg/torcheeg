import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import DDPMTrainer, CDDPMTrainer
from torcheeg.models import BUNet, BCUNet


class DummyDataset(Dataset):
    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(4, 9, 9), random.randint(0, 1)


class TestDDPMTrainer(unittest.TestCase):
    def test_ddpm_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        unet = BUNet(in_channels=4)

        trainer = DDPMTrainer(unet)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = DDPMTrainer(unet, device_ids=[0])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    def test_cddpm_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        unet = BCUNet(in_channels=4, num_classes=2)

        trainer = CDDPMTrainer(unet)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = CDDPMTrainer(unet, device_ids=[0])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
