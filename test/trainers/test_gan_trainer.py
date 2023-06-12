import random
import unittest
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import CWGANGPTrainer, WGANGPTrainer
from torcheeg.models import BGenerator, BDiscriminator, BCGenerator, BCDiscriminator


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
                                   nn.Conv2d(128, 256, kernel_size=4, stride=1),
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
                                   nn.Conv2d(128, 256, kernel_size=4, stride=1),
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


class TestWGANGPTrainer(unittest.TestCase):

    def test_wgan_gp_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        g_model = BGenerator(in_channels=128)
        d_model = BDiscriminator(in_channels=4)

        trainer = WGANGPTrainer(g_model,
                             d_model,
                             metric_extractor=Extractor(),
                             metric_classifier=Classifier(),
                             metric_num_features=9 * 9 * 64,
                             metrics=['fid', 'is'],
                             accelerator='gpu')
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

    def test_cwgan_gp_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        g_model = BCGenerator(in_channels=128)
        d_model = BCDiscriminator(in_channels=4)

        trainer = CWGANGPTrainer(g_model,
                              d_model,
                              metric_extractor=Extractor(),
                              metric_classifier=Classifier(),
                              metric_num_features=9 * 9 * 64,
                              metrics=['fid'],
                             accelerator='gpu')
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
