import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from torch.optim import Optimizer

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger(__name__)


class FinetuningCallback(BaseFinetuning):
    def __init__(self, freeze_epochs: int = 10, freeze_layers: List[str] = []):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.freeze_layers = freeze_layers

    def freeze_before_training(self, pl_module: pl.LightningModule):
        for layer_name in self.freeze_layers:
            module = pl_module
            for attr in layer_name.split('.'):
                module = getattr(module, attr)
            self.freeze(module)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int,
                          optimizer: Optimizer):
        if epoch == self.freeze_epochs:
            # unfreeze remaing layers
            for layer_name in self.freeze_layers:
                module = pl_module
                for attr in layer_name.split('.'):
                    module = getattr(module, attr)
                self.unfreeze_and_add_param_group(module, optimizer=optimizer)


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + \
           (1 - lam) * criterion(pred, y_b)


from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr *
            ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = ['precision', 'recall', 'f1score', 'accuracy']

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'"
            )
    metric_dict = {
        'accuracy':
        torchmetrics.Accuracy(task='multiclass',
                              num_classes=num_classes,
                              top_k=1),
        'precision':
        torchmetrics.Precision(task='multiclass',
                               average='macro',
                               num_classes=num_classes),
        'recall':
        torchmetrics.Recall(task='multiclass',
                            average='macro',
                            num_classes=num_classes),
        'f1score':
        torchmetrics.F1Score(task='multiclass',
                             average='macro',
                             num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class FinetuneTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.

        .. code-block:: python

            trainer = FinetuneTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            mixup_alpha (float): The alpha parameter of the mixup data augmentation. If set to 0.0, no mixup will be used. (default: :obj:`0.2`)
            cosine_scheduler (bool): Whether to use cosine annealing scheduler. (default: :obj:`False`)
            milestones (list of int): The milestones of the learning rate scheduler. (default: :obj:`[100]`)
            freeze_epochs (int): The number of epochs to freeze the layers. (default: :obj:`10`)
            freeze_layers (list of str): The names of the layers to freeze. (default: :obj:`[]`)
            warmup_epochs (int): The number of epochs to warmup the learning rate. (default: :obj:`10`)
            warmup_lr (float): The learning rate to warmup to. (default: :obj:`1e-5`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`["accuracy"]`)
        
        .. automethod:: fit
        .. automethod:: test
    '''
    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 mixup_alpha: float = 0.0,
                 cosine_scheduler: bool = False,
                 milestones: List[int] = [100],
                 freeze_epochs: int = 1,
                 freeze_layers: List[str] = [],
                 warmup_epochs: int = 10,
                 warmup_lr: float = 1e-5,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.mixup_alpha = mixup_alpha
        self.cosine_scheduler = cosine_scheduler
        self.milestones = milestones
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.freeze_epochs = freeze_epochs
        self.freeze_layers = freeze_layers

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()
        self.len_train_loader = 0

        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            callbacks: List[pl.Callback] = [],
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        callbacks += [
            FinetuningCallback(self.freeze_epochs, self.freeze_layers)
        ]
        self.len_train_loader = len(train_loader)
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             callbacks=callbacks,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch

        y_hat = self(x)
        if self.mixup_alpha > 0.0:
            data, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            loss = mixup_criterion(self.ce_fn, self(data), y_a, y_b, lam)
        else:
            loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
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
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.SGD(trainable_parameters,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    momentum=0.9)

        if self.cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs)
        else:
            milestones = [
                (e - self.warmup_epochs) * self.len_train_loader
                for e in self.milestones
            ]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        if self.warmup_epochs > 0:
            total_iter = self.trainer.max_epochs * self.len_train_loader
            multiplier = self.lr / self.warmup_lr
            scheduler = GradualWarmupScheduler(optimizer,
                                               multiplier=multiplier,
                                               total_iter=total_iter,
                                               after_scheduler=scheduler)

        return [optimizer], [scheduler]

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        return y_hat