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
    # def test_glow_trainer(self):
    #     train_dataset = DummyDataset()
    #     val_dataset = DummyDataset()
    #     test_dataset = DummyDataset()

    #     train_loader = DataLoader(train_dataset, batch_size=4)
    #     val_loader = DataLoader(val_dataset, batch_size=4)
    #     test_loader = DataLoader(test_dataset, batch_size=4)

    #     model = BGlow(in_channels=4)

    #     trainer = GlowTrainer(model, accelerator='gpu')
    #     trainer.fit(train_loader, val_loader, max_epochs=1)
    #     trainer.test(test_loader)

    def test_cglow_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=4)

        model = BCGlow(in_channels=4)
        trainer = CGlowTrainer(model, accelerator='gpu')
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
