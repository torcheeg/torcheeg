import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import CORALTrainer, DDCTrainer, DANNTrainer, DANTrainer, ADATrainer


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


class TestDomainAdaptionTrainer(unittest.TestCase):
    def test_coral_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)

        trainer = CORALTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

        trainer = CORALTrainer(extractor,
                               classifier,
                               device_ids=[0])
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)
    
    def test_ada_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)

        trainer = ADATrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

        trainer = ADATrainer(extractor,
                               classifier,
                               device_ids=[0])
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    def test_dan_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)

        trainer = DANTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

        trainer = DANTrainer(extractor,
                               classifier,
                               device_ids=[0])
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)
    
    def test_ddc_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)

        trainer = DDCTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

        trainer = DDCTrainer(extractor,
                               classifier,
                               device_ids=[0])
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    def test_dann_trainer(self):
        source_dataset = DummyDataset()
        target_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        source_loader = DataLoader(source_dataset, batch_size=2)
        target_loader = DataLoader(target_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        extractor = DummyModel(120, 10)
        classifier = DummyModel(10, 2)
        domain_classifier = DummyModel(10, 2)

        trainer = DANNTrainer(extractor, classifier, domain_classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

        trainer = DANNTrainer(extractor,
                               classifier,
                               domain_classifier,
                               device_ids=[0])
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    unittest.main()
