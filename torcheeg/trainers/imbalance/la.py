from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer


class LALoss(nn.Module):

    def __init__(self, class_frequency: List[int], tau=1.0, eps=1e-12):
        '''
        Logit-adjusted (LA) loss for imbalanced datasets.

        - Paper: Menon A K, Jayasumana S, Rawat A S, et al. Long-tail learning via logit adjustment[J]. arXiv preprint arXiv:2007.07314, 2020.
        - URL: https://arxiv.org/abs/2007.07314
        - Related Project: https://github.com/Chumsy0725/logit-adj-pytorch

        Args:
            class_frequency (List[int]): The frequency of each class in the dataset.
            tau (float): The temperature parameter. (default: :obj:`1.0`)
            eps (float): The epsilon parameter. (default: :obj:`1e-12`)
        '''
        super(LALoss, self).__init__()
        class_frequency = torch.tensor(class_frequency)
        self.register_buffer('class_frequency', class_frequency)

        label_probability = class_frequency / class_frequency.sum()
        adjustments = tau * torch.log(label_probability + eps)
        adjustments = adjustments.reshape(1, -1)
        self.register_buffer('adjustments', adjustments.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input += self.adjustments
        return F.cross_entropy(input, target)


class LALossTrainer(ClassifierTrainer):
    r'''
        A trainer class for EEG classification with Logit-adjusted (LA) loss for imbalanced datasets.

        - Paper: Menon A K, Jayasumana S, Rawat A S, et al. Long-tail learning via logit adjustment[J]. arXiv preprint arXiv:2007.07314, 2020.
        - URL: https://arxiv.org/abs/2007.07314
        - Related Project: https://github.com/Chumsy0725/logit-adj-pytorch

        .. code-block:: python

            from torcheeg.models import CCNN
            from torcheeg.trainers import LALossTrainer

            model = CCNN(in_channels=5, num_classes=2)
            trainer = LALossTrainer(model, num_classes=2, class_frequency=[10, 20], tau=1.0)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int): The number of classes in the dataset.
            class_frequency (List[int] or Dataloader): The frequency of each class in the dataset. It can be a list of integers or a dataloader to calculate the frequency of each class in the dataset, traversing the data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc). (default: :obj:`None`)
            tau (float): The temperature parameter. (default: :obj:`1.0`)
            eps (float): The epsilon parameter. (default: :obj:`1e-12`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`["accuracy"]`)
        
        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 class_frequency: Union[List[int], DataLoader],
                 tau: float = 1.0,
                 eps: float = 1e-12,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__(model, num_classes, lr, weight_decay, devices,
                         accelerator, metrics)
        self.tau = tau
        self.eps = eps
        self.class_frequency = class_frequency

        if isinstance(class_frequency, DataLoader):
            _class_frequency = [0] * self.num_classes
            for _, batch_y in class_frequency:
                # assert every item in batch_y is less than self.num_classes
                assert torch.all(batch_y < self.num_classes), f"The label in class_frequency ({batch_y}) is out of range 0-{self.num_classes-1}."

                for y in batch_y:
                    _class_frequency[y] += 1

            self._class_frequency = _class_frequency
        else:
            self._class_frequency = class_frequency

        self.la_fn = LALoss(self._class_frequency, self.tau, self.eps)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.la_fn(y_hat, y)

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

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.la_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.la_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss