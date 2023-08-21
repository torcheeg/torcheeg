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


class SimCLRTrainer(pl.LightningModule):
    r'''
    This class supports the implementation of A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) for self-supervised pre-training.

    - Paper: Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations[C]//International conference on machine learning. PMLR, 2020: 1597-1607.
    - URL: http://proceedings.mlr.press/v119/chen20j.html
    - Related Project: https://github.com/sthalles/SimCLR

    .. code-block:: python

        trainer = SimCLRTrainer(extractor,
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

        trainer = SimCLRTrainer(extractor,
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
        temperature (float): The temperature. (default: :obj:`0.1`)
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
                 temperature: float = 0.1,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["acc_top1"]):
        super().__init__()

        self.extractor = extractor
        self.projector = self.MLP(extract_channels, proj_hid_channels,
                                  proj_channels)
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
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
        xs = torch.cat(xs, dim=0)

        feats = self.extractor(xs)
        feats = self.projector(feats)
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
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None],
             cos_sim.masked_fill(pos_mask, -9e15)
             ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        self.log("train_loss",
                 self.train_loss(nll),
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

        return nll

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

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        xs, _ = batch
        xs = torch.cat(xs, dim=0)

        feats = self.extractor(xs)
        feats = self.projector(feats)
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
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None],
             cos_sim.masked_fill(pos_mask, -9e15)
             ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log("val_loss",
                 self.val_loss(nll),
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

        return nll

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
        optimizer = torch.optim.Adam(
            chain(self.extractor.parameters(), self.projector.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer