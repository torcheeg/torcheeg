from typing import List

import torch
import torch.nn as nn

from .mmd_like import _MMDLikeTrainer


def guassian_kernel(x_source,
                    x_target,
                    mul_kernel=2.0,
                    num_kernels=5,
                    sigma=None):
    n_samples = int(x_source.shape[0]) + int(x_target.shape[0])
    total = torch.cat([x_source, x_target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if sigma:
        bandwidth = sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= mul_kernel**(num_kernels // 2)
    bandwidth_list = [bandwidth * (mul_kernel**i) for i in range(num_kernels)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def maximum_mean_discrepancy(x_source,
                             x_target,
                             mul_kernel=2.0,
                             num_kernels=5,
                             sigma=None):
    batch_size = int(x_source.shape[0])
    kernels = guassian_kernel(x_source,
                              x_target,
                              mul_kernel=mul_kernel,
                              num_kernels=num_kernels,
                              sigma=sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class DANTrainer(_MMDLikeTrainer):
    r'''
    This class supports the implementation of Deep Adaptation Network (DAN) for deep domain adaptation.

    NOTE: DAN belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[C]//International conference on machine learning. PMLR, 2015: 97-105.
    - URL: https://proceedings.mlr.press/v37/long15
    - Related Project: https://github.com/jindongwang/transferlearning/blob/cfaf1174dff7390a861cc4abd5ede37dfa1063f5/code/deep/DAN/DAN.py

    .. code-block:: python

        trainer = DANTrainer(extractor,
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
        weight_domain (float): The weight of the DAN loss. (default: :obj:`1.0`)
        weight_scheduler (bool): Whether to use a scheduler for the weight of the DAN loss, growing from 0 to 1 following the schedule from the DANN paper. (default: :obj:`False`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DANN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the DAN loss is 0. (default: :obj:`0`)
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
        super(DANTrainer, self).__init__(extractor=extractor,
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

        return maximum_mean_discrepancy(x_source_feat, x_target_feat)
