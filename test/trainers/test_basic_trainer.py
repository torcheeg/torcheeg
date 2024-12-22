import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import ClassifierTrainer, RegressorTrainer


class DummyClassificationDataset(Dataset):

    def __init__(self, length: int = 105):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.choice([0, 1])


class DummyRegressionDataset(Dataset):

    def __init__(self, length: int = 105):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(120), random.random()


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestBasicTrainer(unittest.TestCase):

    def test_classifier_trainer(self):
        train_dataset = DummyClassificationDataset()
        val_dataset = DummyClassificationDataset()
        test_dataset = DummyClassificationDataset()

        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        test_loader = DataLoader(test_dataset, batch_size=5)

        model = DummyModel()

        trainer = ClassifierTrainer(model, num_classes=2)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        trainer = ClassifierTrainer(
            model,
            devices=1,
            accelerator='cpu',
            num_classes=2,
            metrics=['accuracy', 'recall', 'precision', 'f1score', 'matthews', 'auroc', 'kappa'])
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        # should catch value error for metrics 'unexpected'
        with self.assertRaises(ValueError):
            trainer = ClassifierTrainer(model,
                                        accelerator='cpu',
                                        num_classes=2,
                                        metrics=['unexpected'])
            trainer.fit(train_loader, val_loader, max_epochs=1)
            trainer.test(test_loader)

    def test_regressor_trainers(self):
        train_dataset = DummyRegressionDataset()
        val_dataset = DummyRegressionDataset()
        test_dataset = DummyRegressionDataset()

        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        test_loader = DataLoader(test_dataset, batch_size=5)

        model = DummyModel(out_channels=1)

        trainer = RegressorTrainer(model)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        trainer = RegressorTrainer(
            model,
            devices=1,
            accelerator='cpu',
            metrics=['mae', 'mse', 'rmse', 'r2score'])
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

        # should catch value error for metrics 'unexpected'
        with self.assertRaises(ValueError):
            trainer = RegressorTrainer(model,
                                        accelerator='cpu',
                                        metrics=['unexpected'])
            trainer.fit(train_loader, val_loader, max_epochs=1)
            trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
