import torch
import pytorch_lightning as pl
import torch.nn as nn

import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Any, Tuple, List


class SimCLRTrainer(pl.LightningModule):
    r'''
    The individual differences and nonstationary nature of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects.This is because the training set and test set come from different data distributions. Domain adaptation is used to address the distribution drift between the training and test sets, thus achieving good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Associative Domain Adaptation (ADA) for deep domain adaptation.

    NOTE: ADA belongs to unsupervised domain adaptation methods, which only use labeled source data and unlabeled target data. This means that the target dataset does not have to contain labels.

    - Paper: Haeusser P, Frerix T, Mordvintsev A, et al. Associative domain adaptation[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2765-2773.
    - URL: https://arxiv.org/abs/1708.00938
    - Related Project: https://github.com/stes/torch-assoc

    .. code-block:: python

        train_set = DEAPDataset(io_path=f'./deap',
                                root_path='./data_preprocessed_python',
                                offline_transform=transforms.Compose([
                                    transforms.BandDifferentialEntropy(
                                        sampling_rate=128, apply_to_baseline=True),
                                    transforms.BaselineRemoval(),
                                    transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                                ]),
                                online_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Contrastive( # see here
                                        transforms.Compose([
                                            RandomNoise(),
                                            RandomMask()
                                        ])
                                        num_views=2)
                                ]),
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]),
                                chunk_size=128,
                                baseline_chunk_size=128,
                                num_baseline=3,
                                num_worker=4)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        trainer = SimCLRTrainer(extractor,
                                     devices=1,
                                     accelerator='gpu')
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        temperature (float): The temperature. (default: :obj:`0.1`)
        devices (int): The number of GPUs to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (List[str]): The metrics to use. Available options are: 'acc_top1', 'acc_top5', 'acc_mean_pos'. (default: :obj:`["acc_top1"]`)

    .. automethod:: fit
    .. automethod:: test
    '''

    def __init__(self,
                 extractor: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 temperature: float = 0.1,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["acc_top1"]):
        super().__init__()

        self.extractor = extractor
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.init_metrics(metrics)

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
        str = "\n[Train] "
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
        optimizer = torch.optim.Adam(self.extractor.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer