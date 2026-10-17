from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.trainers import ClassifierTrainer


class TestClassifierLoss(unittest.TestCase):
    def data(self):
        logits = torch.tensor([[8., -8.], [-2., 2.], [3., -3.], [-4., 4.], [-5., 5.]])
        targets = torch.tensor([0, 1, 0, 1, 0])
        return logits, targets

    def trainer(self):
        return ClassifierTrainer(nn.Identity(), num_classes=2, accelerator='cpu', devices=1, metrics=['accuracy'])

    def test_epoch_loss_weights_unequal_batches_in_all_phases(self):
        logits, targets = self.data()
        expected = F.cross_entropy(logits, targets)
        naive = torch.stack([F.cross_entropy(logits[start:start+2], targets[start:start+2]) for start in range(0,5,2)]).mean()
        self.assertGreater(abs((naive-expected).item()), 1.)
        for phase, step in [('train','training_step'),('val','validation_step'),('test','test_step')]:
            with self.subTest(phase=phase):
                trainer = self.trainer()
                for index, start in enumerate(range(0,5,2)):
                    x = logits[start:start+2].clone().requires_grad_()
                    y = targets[start:start+2]
                    with patch.object(trainer,'log'):
                        loss = getattr(trainer,step)((x,y),index)
                    reference = F.cross_entropy(x,y)
                    torch.testing.assert_close(loss, reference)
                    actual_gradient = torch.autograd.grad(loss,x,retain_graph=True)[0]
                    expected_gradient = torch.autograd.grad(reference,x)[0]
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                torch.testing.assert_close(getattr(trainer,phase+'_loss').compute(), expected)

    def test_equal_size_batches_keep_existing_mean(self):
        logits, targets = self.data()
        trainer = self.trainer()
        for index, start in enumerate((0,2)):
            trainer.test_step((logits[start:start+2],targets[start:start+2]),index)
        torch.testing.assert_close(trainer.test_loss.compute(),F.cross_entropy(logits[:4],targets[:4]))

    def test_real_lightning_test_loss_is_batch_partition_invariant(self):
        logits, targets = self.data()
        expected = F.cross_entropy(logits,targets).item()
        for size in (1,2,4,5):
            with self.subTest(batch_size=size):
                loader = DataLoader(TensorDataset(logits,targets),batch_size=size)
                result = self.trainer().test(loader,logger=False,enable_checkpointing=False,
                    enable_model_summary=False,enable_progress_bar=False)[0]
                self.assertAlmostEqual(result['test_loss'],expected,places=6)
                self.assertAlmostEqual(result['test_accuracy'],.8,places=6)

    def test_epoch_reset_prevents_old_sample_counts_from_leaking(self):
        logits, targets = self.data()
        for phase, step, end in [('train','training_step','on_train_epoch_end'),
                                 ('val','validation_step','on_validation_epoch_end'),
                                 ('test','test_step','on_test_epoch_end')]:
            trainer = self.trainer()
            trainer._trainer = SimpleNamespace(logged_metrics={})
            with self.subTest(phase=phase),patch.object(trainer,'log'):
                getattr(trainer,step)((logits[:4],targets[:4]),0)
                getattr(trainer,end)()
                getattr(trainer,step)((logits[4:],targets[4:]),0)
                torch.testing.assert_close(getattr(trainer,phase+'_loss').compute(),F.cross_entropy(logits[4:],targets[4:]))


if __name__ == '__main__':
    unittest.main()
