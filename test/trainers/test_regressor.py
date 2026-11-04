import random
import unittest
from test.trainers.test_classifier import _TensorDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.trainers import RegressorTrainer


class _DummyRegressionDataset(Dataset):

    def __init__(self, length: int = 105):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        return {'eeg': torch.randn(120), 'label': random.random()}


class _DummyModel(nn.Module):

    def __init__(self, in_channels=120, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class TestRegressorTrainer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_regressor_trainer_basic(self):
        """Test basic functionality of RegressorTrainer"""
        train_dataset = _DummyRegressionDataset()
        val_dataset = _DummyRegressionDataset()
        train_loader = DataLoader(train_dataset, batch_size=5)
        val_loader = DataLoader(val_dataset, batch_size=5)
        model = _DummyModel(out_channels=1)
        trainer = RegressorTrainer(model)
        loss = trainer.training_step(next(iter(train_loader)), batch_idx=0)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.device.type, 'cpu')
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        trainer.validation_step(next(iter(val_loader)), batch_idx=0)

    def test_regressor_trainer_metrics(self):
        """Test metrics configuration of RegressorTrainer"""
        model = _DummyModel(out_channels=1)
        trainer = RegressorTrainer(
            model, metrics=['mae', 'mse', 'rmse', 'r2score'])
        self.assertEqual(trainer.accelerator, 'cpu')
        with self.assertRaises(ValueError):
            RegressorTrainer(model, metrics=['unexpected'])

    def test_cpu_fit_test_and_predict_preserve_batch_shapes_and_metrics(self):
        x = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 48
        y = x.mean(dim=1)
        loader = DataLoader(_TensorDataset(x, y), batch_size=4)
        trainer = RegressorTrainer(nn.Linear(6, 1), metrics=[
                                   'mae', 'mse', 'rmse', 'r2score'])
        options = dict(logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(loader, loader, max_epochs=1, max_steps=1,
                    num_sanity_val_steps=0, **options)
        self.assertEqual(trainer.trainer.global_step, 1)
        results = trainer.test(loader, **options)
        self.assertTrue(all((torch.isfinite(torch.tensor(value))
                        for value in results[0].values())))
        predictions = trainer.predict(loader, **options)
        self.assertEqual(torch.cat(predictions).shape, (8, 1))
        self.assertTrue(torch.isfinite(torch.cat(predictions)).all())


if __name__ == '__main__':
    unittest.main()
