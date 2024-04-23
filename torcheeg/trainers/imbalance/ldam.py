from typing import Any, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer


class LDAMLoss(nn.Module):

    def __init__(self,
                 class_frequency: List[int],
                 max_margin: float = 0.5,
                 weight: Tensor = None,
                 scaling: float = 30):
        '''
        Label-distribution-aware margin (LDAM) loss for imbalanced datasets.

        - Paper: Cao K, Wei C, Gaidon A, et al. Learning imbalanced datasets with label-distribution-aware margin loss[J]. Advances in neural information processing systems, 2019, 32.
        - URL: https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf
        - Related Project: https://github.com/kaidic/LDAM-DRW

        Args:
            class_frequency (List[int]): The frequency of each class in the dataset.
            max_margin (float): The maximum margin. (default: :obj:`0.5`)
            weight (Tensor): The weight of each class. (default: :obj:`None`)
            scaling (float): The scaling factor. (default: :obj:`30`)
        '''
        super(LDAMLoss, self).__init__()
        margin_list = 1.0 / np.sqrt(np.sqrt(class_frequency))
        margin_list = margin_list * (max_margin / np.max(margin_list))
        self.register_buffer('margin_list', torch.tensor(margin_list).float())
        assert scaling > 0, "scaling should be greater than 0."
        self.scaling = scaling
        if not weight is None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        index = torch.zeros_like(input)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        index_bool = index.bool()

        batch_m = torch.matmul(self.margin_list[None, :],
                               index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m

        output = torch.where(index_bool, x_m, input)
        return F.cross_entropy(self.scaling * output,
                               target,
                               weight=self.weight)


class LDAMLossTrainer(ClassifierTrainer):
    r'''
        A trainer class for EEG classification with Label-distribution-aware margin (LDAM) loss for imbalanced datasets.

        - Paper: Cao K, Wei C, Gaidon A, et al. Learning imbalanced datasets with label-distribution-aware margin loss[J]. Advances in neural information processing systems, 2019, 32.
        - URL: https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf
        - Related Project: https://github.com/kaidic/LDAM-DRW

        .. code-block:: python

            from torcheeg.models import CCNN
            from torcheeg.trainers import LDAMLossTrainer

            model = CCNN(in_channels=5, num_classes=2)
            trainer = LDAMLossTrainer(model, num_classes=2, class_frequency=[10, 20], max_margin=0.5, scaling=30)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int): The number of classes in the dataset.
            class_frequency (List[int] or Dataloader): The frequency of each class in the dataset. It can be a list of integers or a dataloader to calculate the frequency of each class in the dataset, traversing the data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc). (default: :obj:`None`)
            max_margin (float): The maximum margin. (default: :obj:`0.5`)
            rule (str): The rule to adjust the weight of each class. Available options are: 'none', 'reweight', 'drw' (deferred re-balancing optimization schedule). (default: :obj:`'none'`)
            beta_reweight (float): The beta parameter for reweighting. It is only used when :obj:`rule` is 'reweight' or 'drw'. (default: :obj:`0.9999`)
            drw_epochs (int): The number of epochs to use DRW. It is only used when :obj:`rule` is 'drw'. (default: :obj:`160`)
            scaling (float): The scaling factor. (default: :obj:`30`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 class_frequency: Union[List[int], DataLoader],
                 max_margin: float = 0.5,
                 scaling: float = 30,
                 rule: str = "none",
                 beta_reweight: float = 0.9999,
                 drw_epochs: int = 160,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):
        super().__init__(model, num_classes, lr, weight_decay, devices,
                         accelerator, metrics)
        self.max_margin = max_margin
        self.scaling = scaling
        self.class_frequency = class_frequency
        self.rule = rule
        self.beta_reweight = beta_reweight
        self.drw_epochs = drw_epochs

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

        assert self.rule in ["none", "reweight",
                             "drw"], f"Unsupported rule: {self.rule}."

        if self.rule == 'none':
            _weight = None
            self._weight = None
        elif self.rule == "reweight":
            effective_num = 1.0 - np.power(self.beta_reweight, self._class_frequency)
            _weight = (1.0 - self.beta_reweight) / np.array(effective_num)
            _weight = _weight / np.sum(_weight) * self.num_classes
            self._weight = torch.tensor(_weight).float()
        else:
            _weight = [1.0] * self.num_classes
            effective_num = 1.0 - np.power(self.beta_reweight, self._class_frequency)
            _drw_weight = (1.0 - self.beta_reweight) / np.array(effective_num)
            _drw_weight = _drw_weight / np.sum(_drw_weight) * self.num_classes
            self._drw_weight = torch.tensor(_drw_weight).float()
            self._weight = torch.tensor(_weight).float()

        self.ldam_fn = LDAMLoss(self._class_frequency, max_margin, self._weight,
                                scaling)

    def on_train_epoch_start(self) -> None:
        # get epoch
        epoch = self.current_epoch
        if epoch == self.drw_epochs and self.rule == "drw":
            # reset the weight buffer in LDAMLoss
            self.ldam_fn = LDAMLoss(self._class_frequency, self.max_margin,
                                    self._drw_weight,
                                    self.scaling).to(self.device)
        return super().on_train_epoch_start()

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ldam_fn(y_hat, y)

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
        loss = self.ldam_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ldam_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss