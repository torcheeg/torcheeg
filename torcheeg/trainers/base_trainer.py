from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from .utils import AverageMeter


class BaseTrainerInterface:
    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        print(f"Epoch {epoch_id}\n-------------------------------")
        self.model = self.model.to(self.device)

    def before_training_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def training_step(self, train_batch: Tuple, batch_id: int,
                      num_batches: int):
        self.model.train()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_id:>5d}/{num_batches:>5d}]")

    def after_training_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def after_training_epoch(self, epoch_id: int, num_epochs: int):
        # optional hooks
        pass

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.model = self.model.to(self.device)
        self.val_loss.reset()
        self.val_correct.reset()

    def before_validation_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def validation_step(self, val_batch: Tuple, batch_id: int,
                        num_batches: int):
        self.model.eval()
        with torch.no_grad():
            X = val_batch[0].to(self.device)
            y = val_batch[1].to(self.device)

            pred = self.model(X)

            self.val_loss.update(self.loss_fn(pred, y).item())
            self.val_correct.update((pred.argmax(1) == y).float().sum().item(), n=X.shape[0])

    def after_validation_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def after_validation_epoch(self, epoch_id: int, num_epochs: int):
        print(
            f"\nTest Error: Avg accuracy: {(100 * self.val_correct.avg):>0.1f}%, Avg loss: {self.val_loss.avg:>8f}"
        )

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        for t in range(num_epochs):
            self.before_training_epoch(t + 1, num_epochs)
            num_batches = len(train_loader)
            for batch_id, train_batch in enumerate(train_loader):
                self.before_training_step(batch_id, num_batches)
                self.training_step(train_batch, batch_id, num_batches)
                self.after_training_step(batch_id, num_batches)
            self.after_training_epoch(t + 1, num_epochs)

            self.before_validation_epoch(t + 1, num_epochs)
            for batch_id, val_batch in enumerate(val_loader):
                num_batches = len(val_loader)
                self.before_validation_step(batch_id, num_batches)
                self.validation_step(val_batch, batch_id, num_batches)
                self.after_validation_step(batch_id, num_batches)
            self.after_validation_epoch(t + 1, num_epochs)

    def score(self, test_loader: DataLoader):
        self.model.eval()
        correct = AverageMeter()
        with torch.no_grad():
            for batch_id, val_batch in enumerate(test_loader):
                X = val_batch[0].to(self.device)
                y = val_batch[1].to(self.device)
                pred = self.model(X)
                correct.update((pred.argmax(1) == y).float().sum().item(), X.shape[0])
        return correct.avg


class BaseTrainer(BaseTrainerInterface):
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device='cpu'):
        self.val_loss = AverageMeter()
        self.val_correct = AverageMeter()

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
