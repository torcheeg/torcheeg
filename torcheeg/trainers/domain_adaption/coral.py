from itertools import chain
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer
from .utils import DualDataLoader


class CORALTrainer(ClassifierTrainer):
    r'''
    The individual differences and nonstationary nature of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects.This is because the training set and test set come from different data distributions. Domain adaptation is used to address the distribution drift between the training and test sets, thus achieving good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of CORrelation ALignment (CORAL) for deep domain adaptation.

    NOTE: CORAL belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Sun B, Saenko K. Deep CORAL: Correlation alignment for deep domain adaptation[C]//European conference on computer vision. Springer, Cham, 2016: 443-450.
    - URL: https://arxiv.org/abs/1607.01719
    - Related Project: https://github.com/adapt-python/adapt/blob/master/adapt/feature_based/_deepCORAL.py

    .. code-block:: python

        trainer = CORALTrainer(extractor,
                             classifier,
                             num_classes=10,
                             devices=1,
                             coral_weight=1.0,
                             accelerator='gpu')
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function. 
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        coral_weight (float): The weight of the CORAL loss. (default: :obj:`1.0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`["accuracy"]`)

    .. automethod:: fit
    .. automethod:: test
    '''

    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 coral_weight: float = 1.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(ClassifierTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.coral_weight = coral_weight
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.init_metrics(metrics, num_classes)

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

        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        batch_size = x_source.shape[0]

        # Compute the loss value
        batch_size = x_source_feat.shape[1]

        # source covariance
        x_source_devi = torch.mean(x_source_feat, 1,
                                   keepdim=True) - x_source_feat
        x_source_cova = torch.matmul(torch.transpose(x_source_devi, 0, 1),
                                     x_source_devi)

        # target covariance
        x_target_devi = torch.mean(x_target_feat, 1,
                                   keepdim=True) - x_target_feat
        x_target_cova = torch.matmul(torch.transpose(x_target_devi, 0, 1),
                                     x_target_devi)
        # frobenius norm between source and target
        coral_loss = torch.mean(
            torch.mul((x_source_cova - x_target_cova),
                      (x_source_cova - x_target_cova))) / (4 * batch_size * 4)

        task_loss = self.ce_fn(y_source_pred, y_source)
        loss = task_loss + self.coral_weight * coral_loss

        self.log("train_loss",
                 self.train_loss(loss),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.extractor.parameters(),
                                           self.classifier.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer