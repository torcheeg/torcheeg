from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmd_like import _MMDLikeTrainer


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


class ADATrainer(_MMDLikeTrainer):
    r'''
    This class supports the implementation of Associative Domain Adaptation (ADA) for deep domain adaptation.

    NOTE: ADA belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Haeusser P, Frerix T, Mordvintsev A, et al. Associative domain adaptation[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2765-2773.
    - URL: https://arxiv.org/abs/1708.00938
    - Related Project: https://github.com/stes/torch-assoc

    .. code-block:: python

        trainer = ADATrainer(extractor,
                            classifier,
                            num_classes=10,
                            devices=1,
                            weight_visit=0.6,
                            accelerator='gpu')
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        weight_walker (float): The weight of the walker loss. (default: :obj:`1.0`)
        weight_visit (float): The weight of the visit loss. (default: :obj:`1.0`)
        weight_domain (float): The weight of the associative loss (default: :obj:`1.0`)
        weight_scheduler (bool): Whether to use a scheduler for the weight of the associative loss, growing from 0 to 1 following the schedule from the DANN paper. (default: :obj:`False`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DANN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the associative loss is 0. (default: :obj:`0`)
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
                 weight_walker: float = 1.0,
                 weight_visit: float = 1.0,
                 weight_domain: float = 1.0,
                 weight_decay: float = 0.0,
                 weight_scheduler: bool = False,
                 lr_scheduler: bool = False,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super(ADATrainer, self).__init__(extractor=extractor,
                                         classifier=classifier,
                                         num_classes=num_classes,
                                         lr=lr,
                                         weight_decay=weight_decay,
                                         weight_domain=weight_domain,
                                         weight_scheduler=weight_scheduler,
                                         lr_scheduler=lr_scheduler,
                                         warmup_epochs=warmup_epochs,
                                         devices=devices,
                                         accelerator=accelerator,
                                         metrics=metrics)
        self.weight_walker = weight_walker
        self.weight_visit = weight_visit

        self._assoc_fn = AssociativeLoss(weight_walker, weight_visit)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)

        domain_loss = self._domain_loss_fn(x_source_feat, x_target_feat,
                                           y_source)

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

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor,
                        y_source: torch.Tensor) -> torch.Tensor:

        return self._assoc_fn(x_source_feat, x_target_feat, y_source)