import logging
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

_EVALUATE_OUTPUT = List[Dict[str, float]]

log = logging.getLogger('torcheeg')


class RootMeanSquaredError(torchmetrics.MeanSquaredError):
    def __init__(self, *args, **kwargs):
        super().__init__(squared=False, *args, **kwargs)


def regression_metrics(metric_list: List[str]):
    allowed_metrics = ['mae', 'mse', 'rmse', 'r2score']

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'mae', 'mse', 'rmse', 'r2score'."
            )
    metric_dict = {
        'mae': torchmetrics.MeanAbsoluteError(),
        'mse': torchmetrics.MeanSquaredError(),
        'rmse': RootMeanSquaredError(),
        'r2score': torchmetrics.R2Score()
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class RegressorTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG regression.

        .. code-block:: python

            trainer = RegressorTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): The regression model that outputs continuous values. The dimension of its output should match the number of target variables to predict.
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'mse' (Mean Squared Error), 'mae' (Mean Absolute Error), 'rmse' (Root Mean Squared Error), 'r2' (R-squared score). (default: :obj:`["mse"]`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["mse"]):

        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.mae_fn = nn.L1Loss()
        self.init_metrics(metrics)

    def init_metrics(self, metrics: List[str]) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = regression_metrics(metrics)
        self.val_metrics = regression_metrics(metrics)
        self.test_metrics = regression_metrics(metrics)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def predict(self, test_loader: DataLoader, *args,
                **kwargs) -> Union[List[Any], List[List[Any]], None]:
        trainer = pl.Trainer(devices=1,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.predict(self, test_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x, *args, **kwargs)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.mae_fn(y_hat[:, 0], y)

        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat[:, 0], y),
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
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str)

        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.mae_fn(y_hat[:, 0], y)
        # print("y_hat", y_hat.shape, y_hat.min(), y_hat.max())
        # print("y", y.shape, y.min(), y.max())

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat[:, 0], y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str)

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.mae_fn(y_hat[:, 0], y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat[:, 0], y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str)

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        return y_hat
