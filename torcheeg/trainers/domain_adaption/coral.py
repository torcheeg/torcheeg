from itertools import cycle, chain
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ..base_trainer import BaseTrainerInterface
from ..utils import AverageMeter


class CoralTrainerInterface(BaseTrainerInterface):
    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        print(f"Epoch {epoch_id}\n-------------------------------")
        self.extractor = self.extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

    def training_step(self, source_loader: DataLoader,
                      target_loader: DataLoader, batch_id: int,
                      num_batches: int):
        self.extractor.train()
        self.classifier.train()

        if self.match_mean:
            match_mean = 1.0
        else:
            match_mean = 0.0

        self.optimizer.zero_grad()

        X_source = source_loader[0].to(self.device)
        y_source = source_loader[1].to(self.device)

        X_target = target_loader[0].to(self.device)

        # Compute prediction error
        X_source_feat = self.extractor(X_source, training=True)
        y_source_pred = self.classifier(X_source_feat, training=True)

        X_target_feat = self.extractor(X_target, training=True)

        batch_size = X_source.shape[0]

        factor_1 = 1. / (batch_size - 1. + np.finfo(np.float32).eps)
        factor_2 = 1. / batch_size

        sum_source = X_source_feat.sum(dim=0)
        sum_source_row = sum_source.reshape(1, -1)
        sum_source_col = sum_source.reshape(-1, 1)
        cov_source = factor_1 * (
            torch.matmul(torch.transpose(X_source_feat), X_source_feat) -
            factor_2 * torch.matmul(sum_source_col, sum_source_row))

        sum_target = X_target_feat.sum(dim=0)
        sum_target_row = sum_target.reshape(1, -1)
        sum_target_col = sum_target.reshape(-1, 1)
        cov_target = factor_1 * (
            torch.matmul(torch.transpose(X_target_feat), X_target_feat) -
            factor_2 * torch.matmul(sum_target_col, sum_target_row))

        mean_source = X_source_feat.mean(dim=0)
        mean_target = X_target_feat.mean(dim=0)

        # Compute the loss value
        task_loss = self.loss_fn(y_source, y_source_pred)
        disc_loss_cov = 0.25 * torch.pow(cov_source - cov_target, 2)
        disc_loss_mean = torch.pow(mean_source - mean_target, 2)

        disc_loss_cov = disc_loss_cov.mean()
        disc_loss_mean = disc_loss_mean.mean()
        disc_loss = self.lambd * (disc_loss_cov + match_mean * disc_loss_mean)

        loss = task_loss + disc_loss

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_id:>5d}/{num_batches:>5d}]")

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.extractor = self.extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.val_loss.reset()
        self.val_correct.reset()

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):

        zip_loader = zip(source_loader, cycle(
            target_loader)) if len(source_loader) > len(target_loader) else zip(
                cycle(source_loader), target_loader)

        for t in range(num_epochs):
            self.before_training_epoch(t + 1, num_epochs)
            num_batches = sum(1 for _ in zip_loader)
            for batch_id, (source_loader,
                           target_loader) in enumerate(zip_loader):
                self.before_training_step(batch_id, num_batches)
                self.training_step(source_loader, target_loader, batch_id,
                                   num_batches)
                self.after_training_step(batch_id, num_batches)
            self.after_training_epoch(t + 1, num_epochs)

            self.before_validation_epoch(t + 1, num_epochs)
            for batch_id, val_batch in enumerate(val_loader):
                num_batches = len(val_loader)
                self.before_validation_step(batch_id, num_batches)
                self.validation_step(val_batch, batch_id, num_batches)
                self.after_validation_step(batch_id, num_batches)
            self.after_validation_epoch(t + 1, num_epochs)

    def validation_step(self, val_batch: Tuple, batch_id: int,
                        num_batches: int):
        self.extractor.eval()
        self.classifier.eval()

        with torch.no_grad():
            X = val_batch[0].to(self.device)
            y = val_batch[1].to(self.device)

            feat = self.extractor(X)
            pred = self.classifier(feat)

            self.val_loss.update(self.loss_fn(pred, y).item())
            self.val_correct.update((pred.argmax(1) == y).float().sum().item(),
                                    n=X.shape[0])

    def score(self, test_loader: DataLoader):
        self.extractor.eval()
        self.classifier.eval()

        correct = AverageMeter()
        with torch.no_grad():
            for batch_id, val_batch in enumerate(test_loader):
                X = val_batch[0].to(self.device)
                y = val_batch[1].to(self.device)
                feat = self.extractor(X)
                pred = self.classifier(feat)
                correct.update((pred.argmax(1) == y).float().sum().item(),
                               n=X.shape[0])
        return correct.avg


class CoralTrainer(CoralTrainerInterface):
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device='cpu'):
        self.val_loss = AverageMeter()
        self.val_correct = AverageMeter()

        self.extractor = extractor
        self.classifier = classifier

        self.optimizer = torch.optim.Adam(chain(extractor.parameters(),
                                                classifier.parameters()),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device