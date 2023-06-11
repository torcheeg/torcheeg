from typing import List, Tuple

import torch
import torch.nn as nn

from .mmd_like import _MMDLikeTrainer


def gaussian_kernel(x_source,
                    x_target,
                    mul_kernel=2.0,
                    num_kernel=5,
                    fix_sigma=None):
    """
    Code from XLearn: computes the full kernel matrix,
    which is less than optimal since we don't use all of it
    with the linear MMD estimate.
    """
    n_samples = int(x_source.size()[0]) + int(x_target.size()[0])
    total = torch.cat([x_source, x_target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= mul_kernel**(num_kernel // 2)
    bandwidth_list = [bandwidth * (mul_kernel**i) for i in range(num_kernel)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def maximum_mean_discrepancy(kernel_values, batch_size):
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernel_values[s1, s2] + kernel_values[t1, t2]
        loss -= kernel_values[s1, t2] + kernel_values[s2, t1]
    return loss / float(batch_size)


class JANTrainer(_MMDLikeTrainer):
    r'''
    This class supports the implementation of Joint Adaptation Networks (JAN) for deep domain adaptation.

    NOTE: JAN belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Long M, Zhu H, Wang J, et al. Deep transfer learning with joint adaptation networks[C]//International conference on machine learning. PMLR, 2017: 2208-2217.
    - URL: http://proceedings.mlr.press/v70/long17a.html
    - Related Project: https://github.com/criteo-research/pytorch-ada

    .. code-block:: python

        trainer = JANTrainer(extractor,
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
        mul_kernel (tuple of float): The kernel multiplier for the Gaussian kernel. (default: :obj:`(2.0, 2.0)`)
        num_kernel (tuple of int): The number of kernels for the Gaussian kernel. (default: :obj:`(5, 1)`)
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
                 weight_decay: float = 0.0,
                 mul_kernel=(2.0, 2.0),
                 num_kernel=(5, 1),
                 weight_domain: float = 1.0,
                 weight_scheduler: bool = False,
                 lr_scheduler: bool = False,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super(JANTrainer, self).__init__(extractor=extractor,
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
        self.mul_kernel = mul_kernel
        self.num_kernel = num_kernel

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        (x_source, y_source), (x_target, _) = batch

        x_source_feat = self.extractor(x_source)
        y_source_pred = self.classifier(x_source_feat)
        x_target_feat = self.extractor(x_target)
        y_target_pred = self.classifier(x_target_feat)

        domain_loss = self._domain_loss_fn(x_source_feat, x_target_feat,
                                           y_source_pred, y_target_pred)

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
                        y_source_pred: torch.Tensor,
                        y_target_pred: torch.Tensor) -> torch.Tensor:

        batch_size = min(len(y_source_pred), len(y_target_pred))
        if len(y_source_pred) != len(y_target_pred):
            # use the smaller batch size
            y_source_pred = y_source_pred[:batch_size]
            y_target_pred = y_target_pred[:batch_size]
            x_source_feat = x_source_feat[:batch_size]
            x_target_feat = x_target_feat[:batch_size]

        softmax_layer = torch.nn.Softmax(dim=-1)
        source_list = [x_source_feat, softmax_layer(y_source_pred)]
        target_list = [x_target_feat, softmax_layer(y_target_pred)]

        joint_kernels = None
        for source, target, k_mul, k_num, sigma in zip(source_list, target_list,
                                                       self.mul_kernel,
                                                       self.num_kernel,
                                                       [None, 1.68]):
            kernels = gaussian_kernel(source,
                                      target,
                                      mul_kernel=k_mul,
                                      num_kernel=k_num,
                                      fix_sigma=sigma)
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return maximum_mean_discrepancy(joint_kernels, batch_size)