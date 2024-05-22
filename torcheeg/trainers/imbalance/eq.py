from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer


class EQLoss(nn.Module):

    def __init__(self,
                 class_frequency: List[int],
                 gamma: float = 0.9,
                 lambd: float = 0.005):
        '''
        Equalization loss for imbalanced datasets.
    
        - Paper: Tan J, Wang C, Li B, et al. Equalization loss for long-tailedobject recognition[C]//Proceedings of the IEEE/CVF conference on computervision and pattern recognition. 2020: 11662-11671.
        - URL: https://openaccess.thecvf.com/content_CVPR_2020/papersTan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.pdf
        - Related Project: https://github.com/tztztztztz/eql.detectron2

        Args:
            class_frequency (List[int]): The frequency of each class in the dataset.
            gamma (float): The gamma parameter. (default: :obj:`0.9`)
            lambd (float): The lambd parameter. (default: :obj:`0.005`)
        '''
        super(EQLoss, self).__init__()
        class_frequency = torch.tensor(class_frequency)
        self.register_buffer('class_frequency', class_frequency)
        self.gamma = gamma
        self.lambd = lambd

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        onehot_class = F.one_hot(target,
                                 num_classes=self.class_frequency.shape[0])
        beta = (torch.rand(onehot_class.size()) < self.gamma).to(input.device)
        w = 1 - beta * self.lambd * (1 - onehot_class)
        p = torch.exp(input) / (
            (w * torch.exp(input)).sum(axis=1, keepdims=True))
        loss = F.nll_loss(torch.log(p), target)
        return loss


class EQLossTrainer(ClassifierTrainer):
    r'''
        A trainer class for EEG classification with Equalization (EQ) loss for imbalanced datasets.
    
        - Paper: Tan J, Wang C, Li B, et al. Equalization loss for long-tailedobject recognition[C]//Proceedings of the IEEE/CVF conference on computervision and pattern recognition. 2020: 11662-11671.
        - URL: https://openaccess.thecvf.com/content_CVPR_2020/papersTan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.pdf
        - Related Project: https://github.com/tztztztztz/eql.detectron2

        .. code-block:: python

            from torcheeg.models import CCNN
            from torcheeg.trainers import EQLossTrainer

            model = CCNN(in_channels=5, num_classes=2)
            trainer = EQLossTrainer(model, num_classes=2, class_frequency=[10, 20], gamma=0.9, lambd=0.005)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int): The number of classes in the dataset.
            class_frequency (List[int] or Dataloader): The frequency of each class in the dataset. It can be a list of integers or a dataloader to calculate the frequency of each class in the dataset, traversing the data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc). (default: :obj:`None`)
            gamma (float): The gamma parameter. (default: :obj:`0.9`)
            lambd (float): The lambd parameter. (default: :obj:`0.005`)
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
                 gamma: float = 0.9,
                 lambd: float = 0.005,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__(model, num_classes, lr, weight_decay, devices,
                         accelerator, metrics)
        self.gamma = gamma
        self.lambd = lambd
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

        self.eq_fn = EQLoss(self._class_frequency, self.gamma, self.lambd)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.eq_fn(y_hat, y)

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
        loss = self.eq_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.eq_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss