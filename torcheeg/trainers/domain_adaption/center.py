import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from numpy import random
from torch.autograd.function import Function

from ..classifier import ClassifierTrainer, classification_metrics

log = logging.getLogger('torcheeg')


class CentersLoss(nn.Module):
    '''
    ClassCentersLoss
    '''

    def __init__(self, num_centers, center_dim, size_average=True):
        super(CentersLoss, self).__init__()
        centers = random.randn(num_centers, center_dim)
        self.centers = nn.Parameter(torch.from_numpy(centers))
        self.ClassCentersfunc = ClassCentersFunc.apply
        self.feat_dim = center_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(
            batch_size if self.size_average else 1)
        loss = self.ClassCentersfunc(feat, label, self.centers,
                                     batch_size_tensor)
        return loss


class ClassCentersFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(
            0,
            label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CenterLossTrainer(ClassifierTrainer):
    r'''
        A trainer trains classification model contains a extractor and a classifier. As for Center loss, it can make the output of the extractor close to the mean of decoded features within the same class. PLease refer to the following infomation to comprehend how the center loss works.

        - Paper: FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery Decoding
        - URL: https://ieeexplore.ieee.org/document/9837422
        - Related Project: https://github.com/Want2Vanish/FBMSNet

        The model structure is required to contains a extractor block which generates the deep feature code and a classifier connected to the extractor to judge which class the feature code belong to.
        Firstly, we should prepare a :obj:`extractor` model and a :obj:`classifier` model for  decoding and classifying from decoding ouput respectly. 
        Here we take FBMSNet as example. :obj:`torcheeg.models.FBMSNet` contains extractor and classifer method already and what We need to do is just to inherit the model to define a extractor and a classifier,and then override the :obj:`forward` method . 

        .. code-block:: python

            from torcheeg.models import FBMSNet
            from torcheeg.trainers import CenterLossTrainer

            class Extractor(FBMSNet):
                def forward(self, x):
                    x = self.mixConv2d(x)
                    x = self.scb(x)
                    x = x.reshape([
                        *x.shape[0:2], self.stride_factor,
                        int(x.shape[3] / self.stride_factor)
                    ])
                    x = self.temporal_layer(x)
                    return torch.flatten(x, start_dim=1)

            class Classifier(FBMSNet):
                def forward(self, x):
                    return self.fc(x)
                
            extractor  = Extractor(num_classes=4,
                                   num_electrodes=22,
                                   chunk_size=512,
                                   in_channels=9)

            classifier = Classifier(num_classes=4,
                                    num_electrodes=22,
                                    chunk_size=512,
                                    in_channels=9)
            
            trainer = CenterLossTrainer(extractor=extractor, 
                                        classifier=classifier,
                                        num_classes=4,
                                        feature_dim=1152)
        
        Args:
            extractor (nn.Module): The extractor which transforms eegsignal into 1D feature code.
            classifier (nn.Module): The classifier that predict from the extractor output which class the siginals belong to.
            feature_dim (int): The dimemsion of extractor output code whose mean values we can loosely regard as the "center". 
            num_classes (int, optional): The number of categories in the dataset. 
            lammda (float): The weight of the center loss in total loss. (default: :obj:`5e-4`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`['accuracy', 'precision', 'recall', 'f1score']`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(
            self,
            extractor,
            classifier,
            feature_dim: int,
            num_classes: int,
            lammda: float = 0.0005,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            devices: int = 1,
            accelerator: str = "cpu",
            metrics: List[str] = ['accuracy', 'precision', 'recall',
                                  'f1score']):

        super(CenterLossTrainer,
              self).__init__(extractor, num_classes, lr, weight_decay, devices,
                             accelerator, metrics)

        self.extractor = extractor
        self.classifier = classifier
        self.center_loss = CentersLoss(num_classes, feature_dim)
        self.lammda = lammda
        self.automatic_optimization = False
        self.feature_dim = feature_dim

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        # for train
        self.train_loss = torchmetrics.MeanMetric()
        self.center_loss_train = torchmetrics.MeanMetric()
        self.predict_loss_train = torchmetrics.MeanMetric()

        # val
        self.val_loss = torchmetrics.MeanMetric()
        self.center_loss_val = torchmetrics.MeanMetric()
        self.predict_loss_val = torchmetrics.MeanMetric()

        #test
        self.test_loss = torchmetrics.MeanMetric()
        self.center_loss_test = torchmetrics.MeanMetric()
        self.predict_loss_test = torchmetrics.MeanMetric()

        # classification metrics for train val test
        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def __reset_metric(self, state: str = "train"):
        if state == "train":
            self.train_loss.reset()
            self.center_loss_train.reset()
            self.predict_loss_train.reset()
            self.train_metrics.reset()
        elif state == "val":
            self.val_loss.reset()
            self.center_loss_val.reset()
            self.predict_loss_val.reset()
            self.val_metrics.reset()
        elif state == "test":
            self.test_loss.reset()
            self.center_loss_test.reset()
            self.predict_loss_test.reset()
            self.test_metrics.reset()

    def training_step(self, batch, batch_idx) -> None:
        x, y = batch
        center_optimizer, model_optimizer = self.optimizers(True)

        # zero_grad
        center_optimizer.zero_grad()
        model_optimizer.zero_grad()

        # center loss
        feat = self.extractor(x)
        centerloss = self.center_loss(y, feat)

        # prediction cross entropy loss
        y_hat = self.classifier(feat)
        pre_loss = self.ce_fn(y_hat, y)

        # total loss
        loss = self.lammda * centerloss + pre_loss

        # backward
        self.manual_backward(loss)

        # step
        center_optimizer.step()
        model_optimizer.step()

        # log
        log_dict = {"train_loss": self.train_loss(loss)}

        for i, metric_value in enumerate(self.train_metrics.values()):
            log_dict[f"train_{self.metrics[i]}"] = metric_value(y_hat, y)
        self.log_dict(log_dict,
                      prog_bar=True,
                      on_epoch=False,
                      logger=False,
                      on_step=True)

    def on_train_epoch_end(self) -> None:

        # log loss
        log_dict = {
            "train_loss": self.train_loss.compute(),
        }
        # log classfication metrics
        for i, metric_value in enumerate(self.train_metrics.values()):
            log_dict[f"train_{self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict, prog_bar=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.__reset_metric()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        # calculate feat y_pred
        feat = self.extractor(x)
        y_hat = self.classifier(feat)

        # get loss (pred_loss,center loss,total_loss)
        pre_loss = self.ce_fn(y_hat, y)
        centerloss = self.center_loss(y, feat)
        loss = pre_loss + self.lammda * centerloss

        # update  metrics
        self.val_loss.update(loss)
        self.center_loss_val.update(centerloss)
        self.predict_loss_val.update(pre_loss)
        self.val_metrics.update(y_hat, y)

        # log loss
        log_dict = {"val_loss": self.val_loss.compute()}
        # log metrics
        for i, metric_value in enumerate(self.val_metrics.values()):
            log_dict[f"val_{self.metrics[i]}"] = metric_value.compute()
            self.log_dict(log_dict,
                          prog_bar=True,
                          on_epoch=True,
                          on_step=False,
                          logger=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        # log loss
        log_dict = {"val_loss": self.val_loss.compute()}
        # log classfication metrics
        for i, metric_value in enumerate(self.val_metrics.values()):
            log_dict[f"val_{self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict,
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False,
                      logger=True)
        # reset the metrics
        self.__reset_metric("val")
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        # centerloss
        feat = self.extractor(x)
        centerloss = self.center_loss(y, feat)

        # predict loss
        y_hat = self.classifier(feat)
        pre_loss = self.ce_fn(y_hat, y)

        #Total loss
        loss = pre_loss + self.lammda * centerloss

        self.test_loss.update(loss)
        self.center_loss_test.update(centerloss)
        self.predict_loss_test(pre_loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        log_dict = {"test_loss": self.test_loss.compute()}
        # log classfication metrics
        for i, metric_value in enumerate(self.test_metrics.values()):
            log_dict[f"test_{self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict,
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False,
                      logger=True)
        # reset the metrics
        self.__reset_metric("test")
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.extractor.parameters())
        parameters.extend(list(self.classifier.parameters()))
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        model_optimizer = torch.optim.Adam(trainable_parameters,
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)

        center_optimizer = torch.optim.SGD(self.center_loss.parameters(),
                                           lr=0.01)
        return center_optimizer, model_optimizer