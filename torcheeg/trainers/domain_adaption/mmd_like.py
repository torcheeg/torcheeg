from itertools import chain
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer, classification_metrics


class DualDataLoader:
    def __init__(self, ref_dataloader: DataLoader,
                 other_dataloader: DataLoader):
        self.ref_dataloader = ref_dataloader
        self.other_dataloader = other_dataloader

    def __iter__(self):
        return self.dual_iterator()

    def __len__(self):
        return len(self.ref_dataloader)

    def dual_iterator(self):
        other_it = iter(self.other_dataloader)
        for data in self.ref_dataloader:
            try:
                data_ = next(other_it)
            except StopIteration:
                other_it = iter(self.other_dataloader)
                data_ = next(other_it)
            yield data, data_


class _MMDLikeTrainer(ClassifierTrainer):
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 weight_domain: float = 1.0,
                 weight_scheduler: bool = False,
                 lr_scheduler: bool = False,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(ClassifierTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier

        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_domain = weight_domain
        self.weight_scheduler = weight_scheduler
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.num_classes = num_classes
        self.metrics = metrics
        self.init_metrics(metrics, num_classes)

        self._ce_fn = nn.CrossEntropyLoss()

        self.num_batches = None  # init in 'fit' method
        self.non_warmup_epochs = None  # init in 'fit' method
        self.lr_factor = 1.0  # update in 'on_train_batch_start' method
        self.weight_factor = 1.0  # update in 'on_train_batch_start' method
        self.scheduled_weight_domain = 1.0  # update in 'on_train_batch_start' method

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_domain_loss = torchmetrics.MeanMetric()
        self.train_task_loss = torchmetrics.MeanMetric()

        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs):
        r'''
        Args:
            source_loader (DataLoader): Iterable DataLoader for traversing the data batch from the source domain (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            target_loader (DataLoader): Iterable DataLoader for traversing the training data batch from the target domain (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc). The target dataset does not have to return labels.
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): The maximum number of epochs to train. (default: :obj:`300`)
        '''

        train_loader = DualDataLoader(source_loader, target_loader)

        self.num_batches = len(train_loader)
        self.non_warmup_epochs = max_epochs - self.warmup_epochs

        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def on_train_batch_start(self, batch, batch_idx):
        if self.current_epoch >= self.warmup_epochs:
            delta_epoch = self.current_epoch - self.warmup_epochs
            p = (batch_idx + delta_epoch * self.num_batches) / (
                self.non_warmup_epochs * self.num_batches)
            self.weight_factor = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            if self.lr_scheduler:
                self.lr_factor = 1.0 / ((1.0 + 10 * p)**0.75)

        if self.weight_scheduler:
            self.scheduled_weight_domain = self.weight_domain * self.weight_factor

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        domain_loss = self._domain_loss_fn(x_source_feat, x_target_feat)

        task_loss = self._ce_fn(y_source_pred, y_source)
        
        if self.current_epoch >= self.warmup_epochs:
            loss = task_loss + self.scheduled_weight_domain * domain_loss
        else:
            loss = task_loss

        self.log("train_domain_loss",
                 self.train_domain_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_task_loss",
                 self.train_task_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_source_pred, y_source),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_domain_loss",
                 self.train_domain_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_task_loss",
                 self.train_task_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.train_domain_loss.reset()
        self.train_task_loss.reset()
        self.train_metrics.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.extractor.parameters(),
                                           self.classifier.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: self.lr_factor)
            return [optimizer], [scheduler]
        return [optimizer]