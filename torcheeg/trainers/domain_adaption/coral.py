from typing import List

import torch
import torch.nn as nn

from .mmd_like import _MMDLikeTrainer


def compute_covariance(data):
    n = data.size(0)

    device = data.device

    ones_vector = torch.ones(n).resize(1, n).to(device=device)

    one_onto_D = torch.mm(ones_vector, data)
    mult_right_terms = torch.mm(one_onto_D.t(), one_onto_D)
    mult_right_terms = torch.div(mult_right_terms, n)

    mult_left_terms = torch.mm(data.t(), data)
    covariance_matrix = 1 / (n - 1) * torch.add(mult_left_terms, -1 *
                                                (mult_right_terms))

    return covariance_matrix


class CORALTrainer(_MMDLikeTrainer):
    r'''
    This class supports the implementation of CORrelation ALignment (CORAL) for deep domain adaptation.

    NOTE: CORAL belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Sun B, Saenko K. Deep CORAL: Correlation alignment for deep domain adaptation[C]//European conference on computer vision. Springer, Cham, 2016: 443-450.
    - URL: https://arxiv.org/abs/1607.01719
    - Related Project: https://github.com/adapt-python/adapt/blob/master/adapt/feature_based/_deepCORAL.py

    .. code-block:: python

        trainer = CORALTrainer(extractor,
                             classifier,
                             num_classes=10,
                             devices=1,
                             weight_domain=1.0,
                             accelerator='gpu')
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function. 
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        weight_domain (float): The weight of the CORAL loss. (default: :obj:`1.0`)
        weight_scheduler (bool): Whether to use a scheduler for the weight of the CORAL loss, growing from 0 to 1 following the schedule from the DANN paper. (default: :obj:`False`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DANN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the CORAL loss is 0. (default: :obj:`0`)
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
                 weight_domain: float = 1.0,
                 weight_scheduler: bool = True,
                 lr_scheduler_gamma: float = 0.0,
                 lr_scheduler_decay: float = 0.75,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(CORALTrainer,
              self).__init__(extractor=extractor,
                             classifier=classifier,
                             num_classes=num_classes,
                             lr=lr,
                             weight_decay=weight_decay,
                             weight_domain=weight_domain,
                             weight_scheduler=weight_scheduler,
                             lr_scheduler_gamma=lr_scheduler_gamma,
                             lr_scheduler_decay=lr_scheduler_decay,
                             warmup_epochs=warmup_epochs,
                             devices=devices,
                             accelerator=accelerator,
                             metrics=metrics)

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor) -> torch.Tensor:

        d = x_source_feat.size(
            1)  # d-dimensional vectors (same for source, target)

        source_covariance = compute_covariance(x_source_feat)
        target_covariance = compute_covariance(x_target_feat)

        # take Frobenius norm (https://pytorch.org/docs/stable/torch.html)
        loss = torch.norm(torch.mul((source_covariance - target_covariance),
                                    (source_covariance - target_covariance)),
                          p="fro")

        # loss = torch.norm(torch.mm((source_covariance-target_covariance),
        # 							(source_covariance-target_covariance)), p="fro")

        loss = loss / (4 * d * d)
        return loss