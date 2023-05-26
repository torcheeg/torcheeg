from itertools import chain
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer


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


class WalkerLoss(nn.Module):

    def forward(self, P_aba, y):
        equality_matrix = torch.eq(y.reshape(-1, 1), y).float()
        p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        p_target.requires_grad = False

        L_walker = F.kl_div(torch.log(1e-8 + P_aba),
                            p_target,
                            size_average=False)
        L_walker /= p_target.shape[0]

        return L_walker


class VisitLoss(nn.Module):

    def forward(self, P_b):
        p_visit = torch.ones([1, P_b.shape[1]]) / float(P_b.shape[1])
        p_visit.requires_grad = False
        p_visit = p_visit.to(P_b.device)
        L_visit = F.kl_div(torch.log(1e-8 + P_b), p_visit, size_average=False)
        L_visit /= p_visit.shape[0]

        return L_visit


class AssociationMatrix(nn.Module):

    def __init__(self):
        super(AssociationMatrix, self).__init__()

    def forward(self, X_source, X_target):
        X_source = X_source.reshape(X_source.shape[0], -1)
        X_target = X_target.reshape(X_target.shape[0], -1)

        W = torch.mm(X_source, X_target.transpose(1, 0))

        P_ab = F.softmax(W, dim=1)
        P_ba = F.softmax(W.transpose(1, 0), dim=1)

        P_aba = P_ab.mm(P_ba)
        P_b = torch.mean(P_ab, dim=0, keepdim=True)

        return P_aba, P_b


class AssociativeLoss(nn.Module):

    def __init__(self, walker_weight=1., visit_weight=1.):
        super(AssociativeLoss, self).__init__()

        self.matrix = AssociationMatrix()
        self.walker = WalkerLoss()
        self.visit = VisitLoss()

        self.walker_weight = walker_weight
        self.visit_weight = visit_weight

    def forward(self, X_source, X_target, y):

        P_aba, P_b = self.matrix(X_source, X_target)
        L_walker = self.walker(P_aba, y)
        L_visit = self.visit(P_b)

        return self.visit_weight * L_visit + self.walker_weight * L_walker


class ADATrainer(ClassifierTrainer):
    r'''
    The individual differences and nonstationary nature of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects.This is because the training set and test set come from different data distributions. Domain adaptation is used to address the distribution drift between the training and test sets, thus achieving good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Associative Domain Adaptation (ADA) for deep domain adaptation.

    NOTE: ADA belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Haeusser P, Frerix T, Mordvintsev A, et al. Associative domain adaptation[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2765-2773.
    - URL: https://arxiv.org/abs/1708.00938
    - Related Project: https://github.com/stes/torch-assoc

    .. code-block:: python

        trainer = ADATrainer(extractor,
                            classifier,
                            num_classes=10,
                            devices=1,
                            visit_weight=0.6,
                            accelerator='gpu')
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        lr_decay (int): The learning rate decay. The authors choose an initial learning rate of 0.0001, which is reduced by a factor of 0.33 in the last third of the training time. :obj:`lr_decay` is used to define the epoch to apply the decay of the learning rate (default: :obj:`200`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        walker_weight (float): The weight of the walker loss. (default: :obj:`1.0`)
        visit_weight (float): The weight of the visit loss. (default: :obj:`1.0`)
        assoc_weight (float): The weight of the associative loss. (default: :obj:`1.0`)
        assoc_delay (int): The delay in applying the associative loss. (default: :obj:`10`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. (default: :obj:`["accuracy"]`)

    .. automethod:: fit
    .. automethod:: test
    '''

    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 lr_decay: int = 200,
                 weight_decay: float = 0.0,
                 walker_weight: float = 1.0,
                 visit_weight: float = 1.0,
                 assoc_weight: float = 1.0,
                 assoc_delay: int = 10,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(ClassifierTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier
        self.num_classes = num_classes
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.walker_weight = walker_weight
        self.visit_weight = visit_weight
        self.assoc_weight = assoc_weight
        self.assoc_delay = assoc_delay
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.assoc_fn = AssociativeLoss(self.walker_weight,
                                                   self.visit_weight)
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

        assoc = self.assoc_fn(x_source_feat,
                                                    x_target_feat, y_source)
        task_loss = self.ce_fn(y_source_pred, y_source)
        # if epoch num is less than delay, only train task loss
        if self.current_epoch < self.assoc_delay:
            loss = task_loss
        else:
            loss = task_loss + self.assoc_weight * assoc

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
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.999),
                                     amsgrad=True)
        # decay lr after self.lr_decay epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_decay,
                                                    gamma=0.33)
        return [optimizer], [scheduler]