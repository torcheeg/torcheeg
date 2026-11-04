import random
import unittest

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torcheeg.models import BCGlow, BGlow
from torcheeg.trainers import CGlowTrainer, GlowTrainer


class _DummyDataset(Dataset):

    def __init__(self, length: int = 101):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        labels = random.randint(0, 1)
        return {'eeg': torch.ones(4, 32, 32) * labels * 0.8 + torch.randn(4, 32, 32) * 0.2, 'label': labels}


class TestGlowTrainer(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(random.setstate, random.getstate())
        random.seed(0)

    def test_glow_fit_runs_optimization_and_reports_finite_metrics(self):
        train_dataset = _DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=4)
        model = BGlow(in_channels=4)
        trainer = GlowTrainer(model, accelerator='cpu')
        trainer.fit(train_loader, train_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all((torch.isfinite(torch.as_tensor(value)).all()
                        for value in trainer.trainer.logged_metrics.values())))
        trainer.test_step(next(iter(train_loader)), batch_idx=0)

    def test_cglow_fit_runs_optimization_and_reports_finite_metrics(self):
        train_dataset = _DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=4)
        model = BCGlow(in_channels=4)
        trainer = CGlowTrainer(model, accelerator='cpu')
        trainer.fit(train_loader, train_loader, max_epochs=1, max_steps=1)
        self.assertGreater(trainer.trainer.global_step, 0)
        self.assertTrue(trainer.trainer.logged_metrics)
        self.assertTrue(all((torch.isfinite(torch.as_tensor(value)).all()
                        for value in trainer.trainer.logged_metrics.values())))
        trainer.test_step(next(iter(train_loader)), batch_idx=0)


if __name__ == '__main__':
    unittest.main()
