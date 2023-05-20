import unittest

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torcheeg.algorithms import ClassificationAlgorithm
import pytorch_lightning as pl


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestClassificationClassificationAlgorithm(unittest.TestCase):

    def test_classification_algorithm(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)
        model = DummyModel()
        algorithm = ClassificationAlgorithm(model, num_classes=2)

        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=10)
        trainer.fit(algorithm, train_loader, val_loader)
        trainer.test(
            algorithm, test_loader
        )  # return [{'test_loss': 0.7485857605934143, 'test_accuracy': 0.49504950642585754}]
        trainer.predict(algorithm, test_loader)


if __name__ == '__main__':
    unittest.main()
