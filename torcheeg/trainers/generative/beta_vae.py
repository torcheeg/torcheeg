import warnings
from itertools import chain
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore

from .utils import FrechetInceptionDistance

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


class BetaVAETrainer(pl.LightningModule):
    r'''
    This class provide the implementation for BetaVAE training. The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the training, test, and new EEG inference of variational autoencoders.

    - Paper: Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of wasserstein gans[J]. Advances in neural information processing systems, 2017, 30.
    - URL: https://arxiv.org/abs/1704.00028
    - Related Project: https://github.com/eriklindernoren/PyTorch-GAN

    .. code-block:: python
        
        encoder = BEncoder(in_channels=4)
        decoder = BDecoder(in_channels=64, out_channels=4)
        trainer = BetaVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        encoder (nn.Module): The encoder, whose inputs are EEG signals, outputs are two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick.
        decoder (nn.Module): The decoder generating EEG signals from hidden variables encoded by the encoder.
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (default: :obj:`0.0`)
        beta: (float): The weight of the KL divergence in the loss function. When beta is 1, the model is a standard VAE. (default: :obj:`1.0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'fid', 'is'. Due to the time-consuming generation process, these indicators will only be calculated and printed during test. (default: :obj:`[]`)
        metric_extractor (nn.Module): The feature extraction model used to calculate the FID and IS metrics. (default: :obj:`None`)
        metric_classifier (nn.Module): The classification model used to calculate the IS metric. (default: :obj:`None`)
        metric_num_features (int): The number of features extracted by the feature extraction model. (default: :obj:`None`)
    
    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 beta: float = 1.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = [],
                 metric_extractor: nn.Module = None,
                 metric_classifier: nn.Module = None,
                 metric_num_features: int = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.mse_fn = nn.MSELoss()

        self.metric_extractor = metric_extractor
        self.metric_classifier = metric_classifier
        self.metric_num_features = metric_num_features
        self.init_metrics(metrics)

    def init_metrics(self, metrics) -> None:
        self.train_rec_loss = torchmetrics.MeanMetric()
        self.val_rec_loss = torchmetrics.MeanMetric()
        self.test_rec_loss = torchmetrics.MeanMetric()

        self.train_kld_loss = torchmetrics.MeanMetric()
        self.val_kld_loss = torchmetrics.MeanMetric()
        self.test_kld_loss = torchmetrics.MeanMetric()

        if 'fid' in metrics:
            assert not self.metric_extractor is None, 'The metric_extractor should be specified.'
            if hasattr(self.metric_extractor,
                       'in_channels') and self.metric_num_features is None:
                warnings.warn(
                    f'No metric_num_features specified, use metric_extractor.in_channels ({self.metric_extractor.in_channels}) as metric_num_features.'
                )
                self.metric_num_features = self.metric_extractor.in_channels
            assert not self.metric_num_features is None, 'The metric_num_features should be specified.'
            self.test_fid = FrechetInceptionDistance(self.metric_extractor,
                                                     self.metric_num_features)

        if 'is' in metrics:
            assert not self.metric_extractor is None, 'The metric_classifier should be specified.'
            self.test_is = InceptionScore(self.metric_classifier)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0,
                     random: bool = True) -> torch.Tensor:
        x, _ = batch
        try:
            mu, log_var = self.encoder(x)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc
        latent = self.reparameterize(mu, log_var)
        # sample from random instead of encoded latent
        if random:
            latent = torch.normal(mean=0, std=1, size=latent.shape).type_as(x)
        return self(latent)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, _ = batch

        try:
            mu, log_var = self.encoder(x)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = rec_loss + self.beta * kld_loss

        self.log("train_rec_loss",
                 self.train_rec_loss(rec_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_kld_loss",
                 self.train_kld_loss(kld_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_rec_loss",
                 self.train_rec_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_kld_loss",
                 self.train_kld_loss.compute(),
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
        self.train_rec_loss.reset()
        self.train_kld_loss.reset()

    @torch.enable_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, _ = batch

        try:
            mu, log_var = self.encoder(x)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = rec_loss + self.beta * kld_loss

        self.val_rec_loss.update(rec_loss)
        self.val_kld_loss.update(kld_loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_rec_loss",
                 self.val_rec_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("val_kld_loss",
                 self.val_kld_loss.compute(),
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
        self.val_rec_loss.reset()
        self.val_kld_loss.reset()

    @torch.enable_grad()
    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, _ = batch

        try:
            mu, log_var = self.encoder(x)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = rec_loss + self.beta * kld_loss

        self.test_rec_loss.update(rec_loss)
        self.test_kld_loss.update(kld_loss)

        # sample from random instead of encoded latent
        latent = torch.normal(mean=0, std=1, size=latent.shape).type_as(x)
        gen_x = self.decoder(latent)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_rec_loss",
                 self.test_rec_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("test_kld_loss",
                 self.test_kld_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        if 'fid' in self.metrics:
            self.log("test_fid",
                     self.test_fid.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if 'is' in self.metrics:
            self.log("test_is",
                     self.test_is.compute()[0],
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[TEST] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.test_rec_loss.reset()
        self.test_kld_loss.reset()

        if 'fid' in self.metrics:
            self.test_fid.reset()
        if 'is' in self.metrics:
            self.test_is.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(chain(self.encoder.parameters(),
                                      self.decoder.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay)


class CBetaVAETrainer(BetaVAETrainer):
    r'''
    This class provide the implementation for BetaVAE training. The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the training, test, and new EEG inference of variational autoencoders.

    - Paper: Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of wasserstein gans[J]. Advances in neural information processing systems, 2017, 30.
    - Paper: Higgins I, Matthey L, Pal A, et al. beta-vae: Learning basic visual concepts with a constrained variational framework[C]//International conference on learning representations. 2017.
    - URL: https://arxiv.org/abs/1704.00028
    - Related Project: https://github.com/eriklindernoren/PyTorch-GAN

    .. code-block:: python
        
        encoder = BCEncoder(in_channels=4, num_classes=2)
        decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=2)
        trainer = CVAETrainer(encoder, decoder)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        encoder (nn.Module): The encoder, whose inputs are EEG signals, outputs are two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick.
        decoder (nn.Module): The decoder generating EEG signals from hidden variables encoded by the encoder. The decoder of CVAE should have an additional input, which is the label of the EEG signal to be generated.
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (default: :obj:`0.0`)
        beta: (float): The weight of the KL divergence in the loss function. When beta is 1, the model is a standard VAE. (default: :obj:`1.0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'fid', 'is'. (default: :obj:`[]`)
        metric_extractor (nn.Module): The feature extraction model used to calculate the FID and IS metrics. (default: :obj:`None`)
        metric_classifier (nn.Module): The classification model used to calculate the IS metric. (default: :obj:`None`)
        metric_num_features (int): The number of features extracted by the feature extraction model. (default: :obj:`None`)
    
    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 beta: float = 1.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = [],
                 metric_extractor: nn.Module = None,
                 metric_classifier: nn.Module = None,
                 metric_num_features: int = None):
        super(CBetaVAETrainer,
              self).__init__(encoder, decoder, lr, weight_decay, beta, devices,
                             accelerator, metrics, metric_extractor,
                             metric_classifier, metric_num_features)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch

        try:
            mu, log_var = self.encoder(x, y)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent, y)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = rec_loss + self.beta * kld_loss

        self.log("train_rec_loss",
                 self.train_rec_loss(rec_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_kld_loss",
                 self.train_kld_loss(kld_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        return loss

    @torch.enable_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch

        try:
            mu, log_var = self.encoder(x, y)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent, y)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = rec_loss + self.beta * kld_loss

        self.val_rec_loss.update(rec_loss)
        self.val_kld_loss.update(kld_loss)

        return loss

    @torch.enable_grad()
    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch

        try:
            mu, log_var = self.encoder(x, y)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc

        latent = self.reparameterize(mu, log_var)
        rec_x = self.decoder(latent, y)

        rec_loss = self.mse_fn(rec_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        self.test_rec_loss.update(rec_loss)
        self.test_kld_loss.update(kld_loss)

        # sample from random instead of encoded latent
        latent = torch.normal(mean=0, std=1, size=latent.shape).type_as(x)
        gen_x = self.decoder(latent, y)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

        return rec_loss, kld_loss

    def forward(self, latent: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, y)

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0,
                     random: bool = True) -> torch.Tensor:
        x, y = batch
        try:
            mu, log_var = self.encoder(x, y)
            # should return two vectors, one for mu, one for log_var
            # otherwise, re-raising the error
        except Exception as exc:
            raise ValueError(
                'The encoder should return two vectors, one for mu, one for log_var.'
            ) from exc
        latent = self.reparameterize(mu, log_var)
        # sample from random instead of encoded latent
        if random:
            latent = torch.normal(mean=0, std=1, size=latent.shape).type_as(x)
        return self(latent, y)