from itertools import chain, cycle
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ..base_trainer import BaseTrainerInterface
from ..utils import AverageMeter


class CoralTrainerInterface(BaseTrainerInterface):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of CORrelation ALignment (CORAL) for deep domain adaptation.

    NOTE: CORAL belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Sun B, Saenko K. Deep coral: Correlation alignment for deep domain adaptation[C]//European conference on computer vision. Springer, Cham, 2016: 443-450.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/adapt-python/adapt/blob/master/adapt/feature_based/_deepcoral.py

    The interface contains the following implementations:

    - :obj:`fit`: used to train the model and validate after each epoch
    - :obj:`score`: used to test the model and output the score of the model on the test set

    The interface provides the following hook functions for inserting additional implementations in the training, validation and testing lifecycle:

    - :obj:`before_training_epoch`: executed before each epoch of training starts
    - :obj:`before_training_step`: executed before each batch of training starts
    - :obj:`training_step`: the training process for each batch
    - :obj:`after_training_step`: execute after the training of each batch
    - :obj:`after_training_epoch`: executed after each epoch of training
    - :obj:`before_validation_epoch`: executed before each round of validation starts
    - :obj:`before_validation_step`: executed before the validation of each batch
    - :obj:`validation_step`: validation process for each batch
    - :obj:`after_validation_step`: executed after the validation of each batch
    - :obj:`after_validation_epoch`: executed after each round of validation

    You can override the methods of this interface to implement your own trainer:

    .. code-block:: python

        class CoralTrainer(CoralTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.extractor = self.extractor.to(self.device)
                self.classifier = self.classifier.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class CoralTrainer(CoralTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                super().before_training_epoch(epoch_id, num_epochs)
        
    '''
    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        print(f"Epoch {epoch_id}\n-------------------------------")
        self.extractor = self.extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

    def training_step(self, source_loader: DataLoader,
                      target_loader: DataLoader, batch_id: int,
                      num_batches: int):
        self.extractor.train()
        self.classifier.train()

        if self.match_mean:
            match_mean = 1.0
        else:
            match_mean = 0.0

        self.optimizer.zero_grad()

        X_source = source_loader[0].to(self.device)
        y_source = source_loader[1].to(self.device)

        X_target = target_loader[0].to(self.device)

        # Compute prediction error
        X_source_feat = self.extractor(X_source, training=True)
        y_source_pred = self.classifier(X_source_feat, training=True)

        X_target_feat = self.extractor(X_target, training=True)

        batch_size = X_source.shape[0]

        factor_1 = 1. / (batch_size - 1. + np.finfo(np.float32).eps)
        factor_2 = 1. / batch_size

        sum_source = X_source_feat.sum(dim=0)
        sum_source_row = sum_source.reshape(1, -1)
        sum_source_col = sum_source.reshape(-1, 1)
        cov_source = factor_1 * (
            torch.matmul(torch.transpose(X_source_feat), X_source_feat) -
            factor_2 * torch.matmul(sum_source_col, sum_source_row))

        sum_target = X_target_feat.sum(dim=0)
        sum_target_row = sum_target.reshape(1, -1)
        sum_target_col = sum_target.reshape(-1, 1)
        cov_target = factor_1 * (
            torch.matmul(torch.transpose(X_target_feat), X_target_feat) -
            factor_2 * torch.matmul(sum_target_col, sum_target_row))

        mean_source = X_source_feat.mean(dim=0)
        mean_target = X_target_feat.mean(dim=0)

        # Compute the loss value
        task_loss = self.loss_fn(y_source, y_source_pred)
        disc_loss_cov = 0.25 * torch.pow(cov_source - cov_target, 2)
        disc_loss_mean = torch.pow(mean_source - mean_target, 2)

        disc_loss_cov = disc_loss_cov.mean()
        disc_loss_mean = disc_loss_mean.mean()
        disc_loss = self.lambd * (disc_loss_cov + match_mean * disc_loss_mean)

        loss = task_loss + disc_loss

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_id:>5d}/{num_batches:>5d}]")

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.extractor = self.extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.val_loss.reset()
        self.val_correct.reset()

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        self._check_dependencies()
        zip_loader = zip(source_loader, cycle(
            target_loader)) if len(source_loader) > len(target_loader) else zip(
                cycle(source_loader), target_loader)

        for t in range(num_epochs):
            self.before_training_epoch(t + 1, num_epochs)
            num_batches = sum(1 for _ in zip_loader)
            for batch_id, (source_loader,
                           target_loader) in enumerate(zip_loader):
                self.before_training_step(batch_id, num_batches)
                self.training_step(source_loader, target_loader, batch_id,
                                   num_batches)
                self.after_training_step(batch_id, num_batches)
            self.after_training_epoch(t + 1, num_epochs)

            self.before_validation_epoch(t + 1, num_epochs)
            for batch_id, val_batch in enumerate(val_loader):
                num_batches = len(val_loader)
                self.before_validation_step(batch_id, num_batches)
                self.validation_step(val_batch, batch_id, num_batches)
                self.after_validation_step(batch_id, num_batches)
            self.after_validation_epoch(t + 1, num_epochs)
        return self

    def validation_step(self, val_batch: Tuple, batch_id: int,
                        num_batches: int):
        self.extractor.eval()
        self.classifier.eval()

        with torch.no_grad():
            X = val_batch[0].to(self.device)
            y = val_batch[1].to(self.device)

            feat = self.extractor(X)
            pred = self.classifier(feat)

            self.val_loss.update(self.loss_fn(pred, y).item())
            self.val_correct.update((pred.argmax(1) == y).float().sum().item(),
                                    n=X.shape[0])

    def score(self, test_loader: DataLoader):
        self._check_dependencies()

        self.extractor.eval()
        self.classifier.eval()

        correct = AverageMeter()
        with torch.no_grad():
            for batch_id, val_batch in enumerate(test_loader):
                X = val_batch[0].to(self.device)
                y = val_batch[1].to(self.device)
                feat = self.extractor(X)
                pred = self.classifier(feat)
                correct.update((pred.argmax(1) == y).float().sum().item(),
                               n=X.shape[0])
        return correct.avg

    def _dependencies(self):
        return {
            'extractor': nn.Module,
            'classifier': nn.Module,
            'optimizer': torch.optim.Optimizer,
            'loss_fn': nn.Module,
            'device': torch.device
        }


class CoralTrainer(CoralTrainerInterface):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of CORrelation ALignment (CORAL) for deep domain adaptation.

    NOTE: CORAL belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Sun B, Saenko K. Deep coral: Correlation alignment for deep domain adaptation[C]//European conference on computer vision. Springer, Cham, 2016: 443-450.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/adapt-python/adapt/blob/master/adapt/feature_based/_deepcoral.py

    .. code-block:: python

        trainer = CoralTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
        score = trainer.score(test_loader)

    The class provides the following hook functions for inserting additional implementations in the training, validation and testing lifecycle:

    - :obj:`before_training_epoch`: executed before each epoch of training starts
    - :obj:`before_training_step`: executed before each batch of training starts
    - :obj:`training_step`: the training process for each batch
    - :obj:`after_training_step`: execute after the training of each batch
    - :obj:`after_training_epoch`: executed after each epoch of training
    - :obj:`before_validation_epoch`: executed before each round of validation starts
    - :obj:`before_validation_step`: executed before the validation of each batch
    - :obj:`validation_step`: validation process for each batch
    - :obj:`after_validation_step`: executed after the validation of each batch
    - :obj:`after_validation_epoch`: executed after each round of validation

    You can override the methods of this interface to implement your own trainer:

    .. code-block:: python

        class CoralTrainer(CoralTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.extractor = self.extractor.to(self.device)
                self.classifier = self.classifier.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class CoralTrainer(CoralTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                super().before_training_epoch(epoch_id, num_epochs)
    
    Args:
        extractor (nn.Module): The feature extraction model, learning the feature representation of EEG signal by forcing the correlation matrixes of source and target data close.
        classifier (nn.Module): The classification model, learning the classification task with source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        device: (torch.device or str): The device on which the model and data is or will be allocated. (defualt: :obj:`False`)
    
    .. automethod:: fit
    .. automethod:: score
    '''
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device=torch.device('cpu')):
        super(CoralTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier

        self.optimizer = torch.optim.Adam(chain(extractor.parameters(),
                                                classifier.parameters()),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        r'''
        Args:
            source_loader (DataLoader): Iterable DataLoader for traversing the data batch from the source domain (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            target_loader (DataLoader): Iterable DataLoader for traversing the training data batch from the target domain (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc). The target dataset does not have to return labels.
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        return super().fit(source_loader=source_loader,
                           target_loader=target_loader,
                           val_loader=val_loader,
                           num_epochs=num_epochs)

    def score(self, test_loader: DataLoader):
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        Returns:
            float: The average classification accuracy of the model on the test set.
        '''
        return super().score(test_loader=test_loader)