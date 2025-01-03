import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.models import BCUNet, BUNet
from torcheeg.trainers import CDDPMTrainer, DDPMTrainer


class DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(4, 9, 9), random.randint(0, 1)


class Extractor(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(
                                       128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        return x


class Classifier(nn.Module):

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(
                                       128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

        self.lin1 = nn.Linear(9 * 9 * 64, 1024)
        self.lin2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class TestDDPMTrainer(unittest.TestCase):
    def setUp(self):
        """Set up common test data"""
        # Prepare minimal datasets and dataloaders for testing
        self.train_dataset = DummyDataset(length=10)  # Reduced dataset size
        self.val_dataset = DummyDataset(length=10)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=4)  # Smaller batch size
        self.val_loader = DataLoader(self.val_dataset, batch_size=4)

    def test_ddpm_trainer_basic(self):
        """Test basic functionality of DDPMTrainer"""
        model = BUNet(in_channels=4)
        trainer = DDPMTrainer(
            model,
            metric_extractor=Extractor(),
            metric_classifier=Classifier(),
            metric_num_features=9 * 9 * 64,
            metrics=['fid', 'is'],
            accelerator='cpu'
        )
        
        # Only test single training step
        trainer.training_step(next(iter(self.train_loader)), batch_idx=0)
        
        # Only test single validation step
        trainer.validation_step(next(iter(self.val_loader)), batch_idx=0)

    def test_cddpm_trainer_basic(self):
        """Test basic functionality of CDDPMTrainer"""
        model = BCUNet(in_channels=4)
        trainer = CDDPMTrainer(
            model,
            metric_extractor=Extractor(),
            metric_classifier=Classifier(),
            metric_num_features=9 * 9 * 64,
            metrics=['fid', 'is'],
            accelerator='cpu'
        )
        
        # Only test single training step
        trainer.training_step(next(iter(self.train_loader)), batch_idx=0)
        
        # Only test single validation step
        trainer.validation_step(next(iter(self.val_loader)), batch_idx=0)


if __name__ == '__main__':
    unittest.main()
