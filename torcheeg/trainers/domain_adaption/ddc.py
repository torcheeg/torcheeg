from typing import List

import torch
import torch.nn as nn

from .mmd_like import _MMDLikeTrainer


def maximum_mean_discrepancy_linear(x_source,
                             x_target):
    delta = x_source - x_target
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


class DDCTrainer(_MMDLikeTrainer):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Deep Domain Confusion (DDC) for deep domain adaptation.

    NOTE: DDC belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. arXiv preprint arXiv:1412.3474, 2014.
    - URL: https://arxiv.org/abs/1412.3474
    - Related Project: https://github.com/syorami/DDC-transfer-learning/blob/master/DDC.py

    .. code-block:: python

        trainer = DDCTrainer(extractor,
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
        weight_domain (float): The weight of the DDC loss. (default: :obj:`1.0`)
        weight_scheduler (bool): Whether to use a scheduler for the weight of the DDC loss, growing from 0 to 1 following the schedule from the DDCN paper. (default: :obj:`False`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DDCN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the DDC loss is 0. (default: :obj:`0`)
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
                 weight_scheduler: bool = False,
                 lr_scheduler: bool = False,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super(DDCTrainer, self).__init__(extractor=extractor,
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

    def _domain_loss_fn(self, x_source_feat: torch.Tensor,
                        x_target_feat: torch.Tensor) -> torch.Tensor:

        batch_size = min(len(x_source_feat), len(x_target_feat))
        if len(x_source_feat) != len(x_target_feat):
            # use the smaller batch size
            x_source_feat = x_source_feat[:batch_size]
            x_target_feat = x_target_feat[:batch_size]

        return maximum_mean_discrepancy_linear(x_source_feat, x_target_feat)
