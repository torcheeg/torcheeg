import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import ClassifierTrainer


class DummyClassificationDataset(Dataset):

    def __init__(self, length: int = 105):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.choice([0, 1])


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestClassifier(unittest.TestCase):
    def test_classifier_trainer_basic(self):
        """Test basic functionality of ClassifierTrainer"""
        # Prepare minimal datasets and dataloaders for testing
        train_dataset = DummyClassificationDataset()
        val_dataset = DummyClassificationDataset()

        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)

        model = DummyModel()
        trainer = ClassifierTrainer(model, num_classes=2)

        # Only test single training step instead of full epoch
        trainer.training_step(next(iter(train_loader)), batch_idx=0)

        # Only test single validation step instead of full epoch
        trainer.validation_step(next(iter(val_loader)), batch_idx=0)

    def test_classifier_trainer_metrics(self):
        """Test metrics configuration of ClassifierTrainer"""
        model = DummyModel()

        # Test if all supported metrics can be initialized properly
        trainer = ClassifierTrainer(
            model,
            num_classes=2,
            metrics=['accuracy', 'recall', 'precision',
                     'f1score', 'matthews', 'auroc', 'kappa']
        )

        # Test if unsupported metrics raise ValueError
        with self.assertRaises(ValueError):
            trainer = ClassifierTrainer(
                model, num_classes=2, metrics=['unexpected'])


if __name__ == '__main__':
    unittest.main()
