import random
import unittest
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import GlowTrainer, CGlowTrainer
from torcheeg.models import BGlow, BCGlow


class DummyDataset(Dataset):
    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        labels = random.randint(0, 1)
        return torch.ones(4, 32, 32) * labels * 0.8 + torch.randn(
            4, 32, 32) * 0.2, labels


class TestGlowTrainer(unittest.TestCase):
    def test_glow_trainer(self):
        train_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=4)

        model = BGlow(in_channels=4)
        trainer = GlowTrainer(model, accelerator='cpu')
        trainer.fit(train_loader, train_loader, max_epochs=1, max_steps=1)
        trainer.test_step(next(iter(train_loader)), batch_idx=0)

    def test_cglow_trainer(self):
        train_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=4)

        model = BCGlow(in_channels=4)
        trainer = CGlowTrainer(model, accelerator='cpu')
        trainer.fit(train_loader, train_loader, max_epochs=1, max_steps=1) 
        trainer.test_step(next(iter(train_loader)), batch_idx=0)

if __name__ == '__main__':
    unittest.main()
