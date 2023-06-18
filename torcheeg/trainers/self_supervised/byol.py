import copy
import logging
from itertools import chain
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class BYOLTrainer(pl.LightningModule):
    r'''
    This class supports the implementation of Bootstrap Your Own Latent (BYOL) for self-supervised pre-training.

    - Paper: Grill J B, Strub F, Altché F, et al. Bootstrap your own latent-a new approach to self-supervised learning[J]. Advances in neural information processing systems, 2020, 33: 21271-21284.
    - URL: https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html
    - Related Project: https://github.com/lucidrains/byol-pytorch

    .. code-block:: python

        trainer = BYOLTrainer(extractor,
                              extract_channels=256,
                              devices=1,
                              accelerator='gpu')
        trainer.fit(train_loader, val_loader)

    NOTE: The first element of each batch in :obj:`train_loader` and :obj:`val_loader` should be a two-tuple, representing two random transformations (views) of data. You can use :obj:`Contrastive` to achieve this functionality.

    .. code-block:: python

        contras_dataset = DEAPDataset(
            io_path=f'./io/deap',
            root_path='./data_preprocessed_python',
            offline_transform=transforms.Compose([
                transforms.BandDifferentialEntropy(sampling_rate=128,
                                                apply_to_baseline=True),
                transforms.BaselineRemoval(),
                transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
            ]),
            online_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Contrastive(transforms.Compose( # see here
                    [transforms.RandomMask(p=0.5),
                    transforms.RandomNoise(p=0.5)]),
                                    num_views=2)
            ]),
            chunk_size=128,
            baseline_chunk_size=128,
            num_baseline=3)

        trainer = BYOLTrainer(extractor,
                              extract_channels=256,
                              devices=1,
                              accelerator='gpu')
        trainer.fit(train_loader, val_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        extract_channels (int): The feature dimensions of the output of the feature extraction model.
        proj_channels (int): The feature dimensions of the output of the projection head. (default: :obj:`256`)
        proj_hid_channels (int): The feature dimensions of the hidden layer of the projection head. (default: :obj:`512`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        devices (int): The number of GPUs to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (List[str]): The metrics to use. Available options are: 'acc_top1', 'acc_top5', 'acc_mean_pos'. (default: :obj:`["acc_top1"]`)

    .. automethod:: fit
    '''
    def __init__(self,
                 extractor: nn.Module,
                 extract_channels: int,
                 proj_channels: int = 256,
                 proj_hid_channels: int = 512,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 moving_average_decay=0.99,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["acc_top1"]):
        super().__init__()

        self.student_model = extractor
        self.student_projector = self.MLP(extract_channels, proj_hid_channels,
                                          proj_channels)
        self.student_predictor = self.MLP(proj_channels, proj_hid_channels,
                                          proj_channels)

        self.teacher_model, self.teacher_projector = self.teacher()

        self.extract_channels = extract_channels
        self.proj_channels = proj_channels
        self.proj_hid_channels = proj_hid_channels

        self.lr = lr
        self.weight_decay = weight_decay

        self.moving_average_decay = moving_average_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.init_metrics(metrics)

    def MLP(self, in_channels: int, hid_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
        )

    def teacher(self) -> Tuple[nn.Module, nn.Module]:
        r'''
        The teacher model is a copy of the student model, but the weights are not updated during training.

        Returns:
            tuple: The teacher model and the projection head.
        '''
        with torch.no_grad():
            teacher_model = copy.deepcopy(self.student_model)
            teacher_projector = copy.deepcopy(self.student_projector)
            return teacher_model, teacher_projector

    def update_moving_average(self) -> None:
        r'''
        Update the weights of the teacher model and the projection head.
        '''
        with torch.no_grad():
            for param_q, param_k in zip(self.student_model.parameters(),
                                        self.teacher_model.parameters()):
                param_k.data = param_k.data * self.moving_average_decay + param_q.data * (
                    1. - self.moving_average_decay)
            for param_q, param_k in zip(self.student_projector.parameters(),
                                        self.teacher_projector.parameters()):
                param_k.data = param_k.data * self.moving_average_decay + param_q.data * (
                    1. - self.moving_average_decay)

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r'''
        The loss function of BYOL.

        Args:
            x (torch.Tensor): The output of the projection head.
            y (torch.Tensor): The output of the projection head of the teacher model.

        Returns:
            torch.Tensor: The loss.
        '''
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def init_metrics(self, metrics) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        if "acc_top1" in metrics:
            self.train_acc_top1 = torchmetrics.MeanMetric()
            self.val_acc_top1 = torchmetrics.MeanMetric()
        if "acc_top5" in metrics:
            self.train_acc_top5 = torchmetrics.MeanMetric()
            self.val_acc_top5 = torchmetrics.MeanMetric()
        if "acc_mean_pos" in metrics:
            self.train_acc_mean_pos = torchmetrics.MeanMetric()
            self.val_acc_mean_pos = torchmetrics.MeanMetric()

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        NOTE: The first element of each batch in :obj:`train_loader` and :obj:`val_loader` should be a two-tuple, representing two random transformations (views) of data. You can use :obj:`Contrastive` to achieve this functionality.

        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        xs, _ = batch

        assert len(xs) == 2, "The number of views must be two in BYOL."

        for i, x in enumerate(xs):
            # # batch size must greater than one
            # assert x.shape[
            #     0] > 1, "Batch size must greater than one, due to the batch normalization layer in the projection head (BYOL)."
            # copy the first element of the batch make the batch size greater than one
            if x.shape[0] == 1:
                xs[i] = torch.cat([x, x[0].unsqueeze(0)], dim=0)

        # Student model
        eeg_one, eeg_two = xs

        student_proj_one = self.student_model(eeg_one)
        student_proj_one = self.student_projector(student_proj_one)

        student_proj_two = self.student_model(eeg_two)
        student_proj_two = self.student_projector(student_proj_two)

        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # Teacher model
            teacher_proj_one = self.teacher_model(eeg_one)
            teacher_proj_one = self.teacher_projector(teacher_proj_one)

            teacher_proj_two = self.teacher_model(eeg_two)
            teacher_proj_two = self.teacher_projector(teacher_proj_two)

        loss_one = self.loss_fn(student_pred_one, teacher_proj_two)
        loss_two = self.loss_fn(student_pred_two, teacher_proj_one)
        loss = (loss_one + loss_two).mean()

        # Get ranking position of positive example
        xs = torch.cat(xs, dim=0)
        feats = self.student_model(xs)
        cos_sim = F.cosine_similarity(feats[:, None, :],
                                      feats[None, :, :],
                                      dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0],
                              dtype=torch.bool,
                              device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None],
             cos_sim.masked_fill(pos_mask, -9e15)
             ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        if "acc_top1" in self.metrics:
            # Logging ranking metrics
            self.log("train_acc_top1",
                     self.train_acc_top1((sim_argsort == 0).float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)
        if "acc_top5" in self.metrics:
            self.log("train_acc_top5",
                     self.train_acc_top5((sim_argsort < 5).float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)
        if "acc_mean_pos" in self.metrics:
            self.log("train_acc_mean_pos",
                     self.train_acc_mean_pos(1 + sim_argsort.float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        if "acc_top1" in self.metrics:
            self.log("train_acc_top1",
                     self.train_acc_top1.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if "acc_top5" in self.metrics:
            self.log("train_acc_top5",
                     self.train_acc_top5.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if "acc_mean_pos" in self.metrics:
            self.log("train_acc_mean_pos",
                     self.train_acc_mean_pos.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        if "acc_top1" in self.metrics:
            self.train_acc_top1.reset()
        if "acc_top5" in self.metrics:
            self.train_acc_top5.reset()
        if "acc_mean_pos" in self.metrics:
            self.train_acc_mean_pos.reset()

    def on_after_backward(self) -> None:
        self.update_moving_average()
        return super().on_after_backward()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        xs, _ = batch

        assert len(xs) == 2, "The number of views must be two in BYOL."
        for i, x in enumerate(xs):
            # # batch size must greater than one
            # assert x.shape[
            #     0] > 1, "Batch size must greater than one, due to the batch normalization layer in the projection head (BYOL)."
            # copy the first element of the batch make the batch size greater than one
            if x.shape[0] == 1:
                xs[i] = torch.cat([x, x[0].unsqueeze(0)], dim=0)

        # Student model
        eeg_one, eeg_two = xs

        student_proj_one = self.student_model(eeg_one)
        student_proj_one = self.student_projector(student_proj_one)

        student_proj_two = self.student_model(eeg_two)
        student_proj_two = self.student_projector(student_proj_two)

        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # Teacher model
            teacher_proj_one = self.teacher_model(eeg_one)
            teacher_proj_one = self.teacher_projector(teacher_proj_one)

            teacher_proj_two = self.teacher_model(eeg_two)
            teacher_proj_two = self.teacher_projector(teacher_proj_two)

        loss_one = self.loss_fn(student_pred_one, teacher_proj_two)
        loss_two = self.loss_fn(student_pred_two, teacher_proj_one)
        loss = (loss_one + loss_two).mean()

        # Get ranking position of positive example
        xs = torch.cat(xs, dim=0)
        feats = self.student_model(xs)
        cos_sim = F.cosine_similarity(feats[:, None, :],
                                      feats[None, :, :],
                                      dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0],
                              dtype=torch.bool,
                              device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None],
             cos_sim.masked_fill(pos_mask, -9e15)
             ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log("val_loss",
                 self.val_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        if "acc_top1" in self.metrics:
            # Logging ranking metrics
            self.log("val_acc_top1",
                     self.val_acc_top1((sim_argsort == 0).float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)
        if "acc_top5" in self.metrics:
            self.log("val_acc_top5",
                     self.val_acc_top5((sim_argsort < 5).float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)
        if "acc_mean_pos" in self.metrics:
            self.log("val_acc_mean_pos",
                     self.val_acc_mean_pos(1 + sim_argsort.float()),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        if "acc_top1" in self.metrics:
            self.log("val_acc_top1",
                     self.val_acc_top1.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if "acc_top5" in self.metrics:
            self.log("val_acc_top5",
                     self.val_acc_top5.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if "acc_mean_pos" in self.metrics:
            self.log("val_acc_mean_pos",
                     self.val_acc_mean_pos.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[VAL] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.val_loss.reset()
        if "acc_top1" in self.metrics:
            self.val_acc_top1.reset()
        if "acc_top5" in self.metrics:
            self.val_acc_top5.reset()
        if "acc_mean_pos" in self.metrics:
            self.val_acc_mean_pos.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.student_model.parameters(),
                                           self.student_projector.parameters(),
                                           self.student_predictor.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer