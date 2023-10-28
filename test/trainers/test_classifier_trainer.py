import random
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torcheeg.trainers import ClassifierTrainer,CLossClassifierTrainer


class DummyDataset(Dataset):

    def __init__(self, length: int = 101, num_class: int = 2 ,data_dim:int =120):
        self.length = length
        self.num_class = num_class
        self.data_dim = data_dim

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return torch.randn(self.data_dim), random.choice(range(self.num_class))


class DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestClassificationTrainer(unittest.TestCase):

    def test_classification_trainer(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = DummyModel()

        trainer = ClassifierTrainer(model, num_classes=2)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        trainer = ClassifierTrainer(
            model,
            devices=1,
            accelerator='cpu',
            num_classes=2,
            metrics=['accuracy', 'recall', 'precision', 'f1score'])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

        # should catch value error for metrics 'unexpected'
        with self.assertRaises(ValueError):
            trainer = ClassifierTrainer(model,
                                        accelerator='cpu',
                                        num_classes=2,
                                        metrics=['unexpected'])
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        trainer = ClassifierTrainer(
            model,
            devices=1,
            accelerator='gpu',
            num_classes=2,
            metrics=['accuracy', 'recall', 'precision', 'f1score'])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)
        
        # test whether the the attribute num_classes is updated when 
        # passing the datasets to .fit() method   
        trainer = ClassifierTrainer(model)
        trainer.fit(train_loader,val_loader,max_epochs=10)
        self.assertEqual(trainer.num_classes,2)
        trainer.test(test_loader)

        # test whether the  attribute num_classes is updated when 
        # passing the datasets to .test() method   
        trainer = ClassifierTrainer(model)
        trainer.test(test_loader)
        self.assertEqual(trainer.num_classes,2)
    
    def test_clossclassification_trainer(self):
        # num_class = 5
        train_dataset = DummyDataset(length=20,num_class=5,data_dim=10)
        val_dataset = DummyDataset(length=20,num_class=5,data_dim=10)
        test_dataset = DummyDataset(length=20,num_class=5,data_dim=10)

        train_loader = DataLoader(train_dataset, batch_size=10)
        val_loader = DataLoader(val_dataset, batch_size=10)
        test_loader = DataLoader(test_dataset, batch_size=10)

        
        model = nn.ModuleDict({"decoder":nn.Linear(10,8),"predict_by_feature":nn.Linear(8,5)})


        trainer = CLossClassifierTrainer(model)
        trainer.fit(train_loader, val_loader,max_epochs=2)
        trainer.test(test_loader)

        trainer = CLossClassifierTrainer(
            model,
            devices=1,
            accelerator='cpu',
            num_classes=5,
            metrics=['accuracy', 'recall', 'precision', 'f1score'])
        
        trainer.fit(train_loader, val_loader,max_epochs=2)
        trainer.test(test_loader)

        # should catch value error for metrics 'unexpected'
        with self.assertRaises(ValueError):
            trainer = CLossClassifierTrainer(model,
                                        accelerator='cpu',
                                        num_classes=5,
                                        metrics=['unexpected'])
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        trainer = CLossClassifierTrainer(
            model,
            devices=1,
            accelerator='gpu',
            num_classes=2,
            metrics=['accuracy', 'recall', 'precision', 'f1score'])
        trainer.fit(train_loader, val_loader,max_epochs=2)
        trainer.test(test_loader)
        
        # num_class = 2
        train_dataset = DummyDataset(length=20,num_class=2,data_dim=10)
        val_dataset = DummyDataset(length=20,num_class=2,data_dim=10)
        test_dataset = DummyDataset(length=20,num_class=2,data_dim=10)

        train_loader = DataLoader(train_dataset, batch_size=10)
        val_loader = DataLoader(val_dataset, batch_size=10)
        test_loader = DataLoader(test_dataset, batch_size=10)

        model = nn.ModuleDict({"decoder":nn.Linear(10,8),"predict_by_feature":nn.Linear(8,2)})

        trainer = CLossClassifierTrainer(model)
        self.assertEqual(trainer.num_classes, None)
        self.assertEqual(trainer.center_dim,  None)
        trainer.fit(train_loader,val_loader,max_epochs=2)
        self.assertEqual(trainer.num_classes,2)
        self.assertEqual(trainer.center_dim,8)
    
        trainer = CLossClassifierTrainer(model)
        trainer.test(test_loader)
        self.assertEqual(trainer.num_classes,2)
        self.assertEqual(trainer.center_dim,8)

        # asign center_dim, not center dim 
        trainer = CLossClassifierTrainer(model,center_dim=8)
        self.assertEqual(trainer.center_dim,8)
        trainer.test(test_loader)
        self.assertEqual(trainer.num_classes,2)
        self.assertEqual(trainer.center_dim,8)

        
        
        


if __name__ == '__main__':
    unittest.main()
