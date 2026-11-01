import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.models import (BCDiscriminator, BCGenerator, BDiscriminator,
                             BGenerator)
from torcheeg.trainers import CWGANGPTrainer, WGANGPTrainer


class _DummyDataset(Dataset):
    def __init__(self, length: int = 50):  # Reduced dataset size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        # Generate random 4-channel 9x9 images and binary labels
        return torch.randn(4, 9, 9), random.randint(0, 1)


class _Extractor(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # 2-layer CNN instead of 4 layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten features for metric computation
        x = x.flatten(start_dim=1)
        return x


class _Classifier(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        # 2-layer CNN + 1 linear layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU())

        # Single linear layer instead of two
        self.lin1 = nn.Linear(64 * 9 * 9, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        return x


class TestWGANGPTrainer(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_wgan_gp_fit_runs_optimization_and_reports_finite_metrics(self):
        # Create minimal datasets
        train_dataset = _DummyDataset()
        val_dataset = _DummyDataset()
        test_dataset = _DummyDataset()

        # Reduced batch size for faster training
        train_loader = DataLoader(train_dataset, batch_size=32)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Initialize models with smaller dimensions
        g_model = BGenerator(in_channels=64)  # Reduced latent dimension
        d_model = BDiscriminator(in_channels=4)

        # Setup trainer with minimal metrics
        trainer = WGANGPTrainer(g_model,
                                d_model,
                                metric_extractor=_Extractor(),
                                metric_classifier=_Classifier(),
                                metric_num_features=64 * 9 * 9,
                                metrics=['fid'],  # Only FID metric
                                accelerator='cpu')
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test_step(next(iter(test_loader)), batch_idx=0)

    def test_cwgan_gp_fit_runs_optimization_and_reports_finite_metrics(self):
        # Same setup for conditional WGAN-GP
        train_dataset = _DummyDataset()
        val_dataset = _DummyDataset()
        test_dataset = _DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=32)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        g_model = BCGenerator(in_channels=64)
        d_model = BCDiscriminator(in_channels=4)

        trainer = CWGANGPTrainer(g_model,
                                 d_model,
                                 metric_extractor=_Extractor(),
                                 metric_classifier=_Classifier(),
                                 metric_num_features=64 * 9 * 9,
                                 metrics=['fid'],
                                 accelerator='cpu')
        trainer.fit(train_loader, val_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                            for value in trainer.trainer.logged_metrics.values()))
        trainer.test_step(next(iter(test_loader)), batch_idx=0)


if __name__ == '__main__':
    unittest.main()
