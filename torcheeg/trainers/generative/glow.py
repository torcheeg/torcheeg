import warnings
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore

from .utils import FrechetInceptionDistance

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


class GlowTrainer(pl.LightningModule):
    r'''
    This class implement the training, test, and new EEG inference of Glow. Glow is dedicated to train an encoder that encodes the input as a hidden variable and makes the hidden variable obey the standard normal distribution. By good design, the encoder should be reversible. On this basis, as soon as the encoder is trained, the corresponding decoder can be used to generate samples from a Gaussian distribution according to the inverse operation. In particular, compared with vanilla normalizing flow-based models, the Glow model is a easy-to-use flow-based model that replaces the operation of permutating the channel axes by introducing a 1x1 reversible convolution.

    - Paper: Kingma D P, Dhariwal P. Glow: Generative flow with invertible 1x1 convolutions[J]. Advances in neural information processing systems, 2018, 31.
    - URL: https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html
    - Related Project: https://github.com/chaiyujin/glow-pytorch

    .. code-block:: python
        
        model = BGlow(in_channels=4)
        trainer = GlowTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        model (nn.Module): Normalized flow model, it needs to implement two interfaces, log_probs and sample. Among them, log_probs takes the original sample as input to calculate the log probs to the target distribution, and sample takes num and temperature as input to calculate the generated sample.
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (default: :obj:`0.0`)
        temperature (float): The temperature. (default: :obj:`1.0`)
        max_grad_clip (float): The maximum norm of the gradients will be clipped to this value. If set to 0, no clipping will be performed. (default: :obj:`0.0`)
        max_grad_norm (float): The maximum norm of the gradients will be normalized to this value. If set to 0, no normalization will be performed. (default: :obj:`0.0`)
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
                 model: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 temperature: float = 1.0,
                 max_grad_clip: float = 0.0,
                 max_grad_norm: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = [],
                 metric_extractor: nn.Module = None,
                 metric_classifier: nn.Module = None,
                 metric_num_features: int = None):
        super().__init__()
        self.model = model

        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.max_grad_clip = max_grad_clip
        self.max_grad_norm = max_grad_norm

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.metric_extractor = metric_extractor
        self.metric_classifier = metric_classifier
        self.metric_num_features = metric_num_features
        self.init_metrics(metrics)

    def on_before_zero_grad(self, optimizer):
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                            self.max_grad_clip)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_grad_norm)

    def init_metrics(self, metrics) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

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
                             num_sanity_val_steps=0,
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
                             num_sanity_val_steps=0,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, num: int, temperature: float = 1.0) -> torch.Tensor:
        return self.model.sample(num, temperature)

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0) -> torch.Tensor:
        x, _ = batch
        return self(x.shape[0], self.temperature)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, _ = batch

        log_probs = self.model.log_probs(x)
        loss = log_probs.mean()

        self.log("train_loss",
                 self.train_loss(loss),
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

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.train_loss.reset()

    @torch.enable_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, _ = batch

        log_probs = self.model.log_probs(x)
        loss = log_probs.mean()
        self.val_loss.update(loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
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

    @torch.enable_grad()
    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, _ = batch

        log_probs = self.model.log_probs(x)
        loss = log_probs.mean()

        self.test_loss.update(loss)

        # sample from random instead of encoded latent
        gen_x = self(x.shape[0], self.temperature)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
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
        self.test_loss.reset()

        if 'fid' in self.metrics:
            self.test_fid.reset()
        if 'is' in self.metrics:
            self.test_is.reset()

    def configure_optimizers(self):
        return torch.optim.Adamax(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)


class CGlowTrainer(GlowTrainer):
    r'''
    This class implement the training, test, and new EEG inference of conditional Glow, which allows the model to generate EEG samples conditioned on the labels. Here, Glow is dedicated to train an encoder that encodes the input as a hidden variable and makes the hidden variable obey the standard normal distribution. By good design, the encoder should be reversible. On this basis, as soon as the encoder is trained, the corresponding decoder can be used to generate samples from a Gaussian distribution according to the inverse operation. In particular, compared with vanilla normalizing flow-based models, the Glow model is a easy-to-use flow-based model that replaces the operation of permutating the channel axes by introducing a 1x1 reversible convolution.

    - Paper: Kingma D P, Dhariwal P. Glow: Generative flow with invertible 1x1 convolutions[J]. Advances in neural information processing systems, 2018, 31.
    - URL: https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html
    - Related Project: https://github.com/chaiyujin/glow-pytorch

    .. code-block:: python
        
        model = BCGlow(in_channels=4)
        trainer = CGlowTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Args:
        model (nn.Module): Normalized flow model, it needs to implement two interfaces, log_probs and sample. Among them, log_probs takes the original sample and label as input, and calculates the log probs and logits of the predicted category to the target distribution, and sample takes label and temperature as input, and calculates the generated sample.
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (default: :obj:`0.0`)
        temperature (float): The temperature. (default: :obj:`1.0`)
        weight_class (float): The weight of the classification loss. (default: :obj:`1.0`)
        max_grad_clip (float): The maximum norm of the gradients will be clipped to this value. If set to 0, no clipping will be performed. (default: :obj:`0.0`)
        max_grad_norm (float): The maximum norm of the gradients will be normalized to this value. If set to 0, no normalization will be performed. (default: :obj:`0.0`)
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
                 model: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 temperature: float = 1.0,
                 weight_class: float = 1.0,
                 max_grad_clip: float = 0.0,
                 max_grad_norm: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = [],
                 metric_extractor: nn.Module = None,
                 metric_classifier: nn.Module = None,
                 metric_num_features: int = None):
        super(CGlowTrainer,
              self).__init__(model=model,
                             lr=lr,
                             weight_decay=weight_decay,
                             temperature=temperature,
                             max_grad_clip=max_grad_clip,
                             max_grad_norm=max_grad_norm,
                             devices=devices,
                             accelerator=accelerator,
                             metrics=metrics,
                             metric_extractor=metric_extractor,
                             metric_classifier=metric_classifier,
                             metric_num_features=metric_num_features)
        self.weight_class = weight_class
        self.ce_fn = nn.CrossEntropyLoss()

    def init_metrics(self, metrics) -> None:
        self.train_kld_loss = torchmetrics.MeanMetric()
        self.val_kld_loss = torchmetrics.MeanMetric()
        self.test_kld_loss = torchmetrics.MeanMetric()

        self.train_class_loss = torchmetrics.MeanMetric()
        self.val_class_loss = torchmetrics.MeanMetric()
        self.test_class_loss = torchmetrics.MeanMetric()

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

    def forward(self,
                y: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        return self.model.sample(y, temperature)

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0) -> torch.Tensor:
        x, y = batch
        return self(y, self.temperature)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch

        kld_loss, y_logits = self.model.log_probs(x, y)
        kld_loss = kld_loss.mean()
        class_loss = self.ce_fn(y_logits, y)
        loss = kld_loss + self.weight_class * class_loss

        self.log("train_kld_loss",
                 self.train_kld_loss(kld_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_class_loss",
                 self.train_class_loss(class_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_kld_loss",
                 self.train_kld_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_class_loss",
                 self.train_class_loss.compute(),
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
        self.train_kld_loss.reset()
        self.train_class_loss.reset()

    @torch.enable_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch

        kld_loss, y_logits = self.model.log_probs(x, y)
        kld_loss = kld_loss.mean()
        class_loss = self.ce_fn(y_logits, y)
        loss = kld_loss + self.weight_class * class_loss

        self.val_kld_loss.update(kld_loss)
        self.val_class_loss.update(class_loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_kld_loss",
                 self.val_kld_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("val_class_loss",
                 self.val_class_loss.compute(),
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
        self.val_kld_loss.reset()
        self.val_class_loss.reset()

    @torch.enable_grad()
    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch

        kld_loss, y_logits = self.model.log_probs(x, y)
        kld_loss = kld_loss.mean()
        class_loss = self.ce_fn(y_logits, y)
        loss = kld_loss + self.weight_class * class_loss

        self.test_kld_loss.update(kld_loss)
        self.test_class_loss.update(class_loss)

        # sample from random instead of encoded latent
        gen_x = self(y, self.temperature)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_kld_loss",
                 self.test_kld_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("test_class_loss",
                 self.test_class_loss.compute(),
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
        self.test_kld_loss.reset()
        self.test_class_loss.reset()

        if 'fid' in self.metrics:
            self.test_fid.reset()
        if 'is' in self.metrics:
            self.test_is.reset()