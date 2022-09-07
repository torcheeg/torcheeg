from itertools import chain, cycle
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data.dataloader import DataLoader

from ..base_trainer import BaseTrainerInterface
from ..utils import AverageMeter


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNTrainerInterface(BaseTrainerInterface):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Domain Adversarial Neural Networks (DANN) for deep domain adaptation.

    NOTE: DANN belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. The journal of machine learning research, 2016, 17(1): 2096-2030.
    - URL: https://arxiv.org/abs/1505.07818
    - Related Project: https://github.com/fungtion/DANN/blob/master/train/main.py

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

        class DANNTrainer(DANNTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.extractor = self.extractor.to(self.device)
                self.classifier = self.classifier.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class DANNTrainer(DANNTrainerInterface):
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
                      num_batches: int, epoch_id: int, num_epochs: int):

        self.extractor.train()
        self.classifier.train()
        self.domain_classifier.train()

        p = float(batch_id + epoch_id * num_batches) / num_epochs / num_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        self.optimizer.zero_grad()

        X_source = source_loader[0].to(self.device)
        y_source = source_loader[1].to(self.device)
        X_target = target_loader[0].to(self.device)

        X_source_feat = self.extractor(X_source)
        y_source_pred = self.classifier(X_source_feat)
        X_source_rev = GradientReverse.apply(X_source_feat, alpha)
        y_source_disc = self.domain_classifier(X_source_rev)

        X_target_feat = self.extractor(X_target)
        X_target_rev = GradientReverse.apply(X_target_feat, alpha)
        y_target_disc = self.domain_classifier(X_target_rev)

        # Compute the loss value
        task_loss = self.loss_fn(y_source, y_source_pred)
        disc_loss_source = self.loss_fn(
            y_source_disc,
            torch.zeros(len(y_source_disc),
                        dtype=torch.long,
                        device=X_source.device))
        disc_loss_target = self.loss_fn(
            y_target_disc,
            torch.ones(len(y_target_disc),
                       dtype=torch.long,
                       device=X_target.device))
        disc_loss = self.lambd * (disc_loss_source + disc_loss_target)

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

        for epoch_id in range(num_epochs):
            self.before_training_epoch(epoch_id + 1, num_epochs)
            num_batches = sum(1 for _ in zip_loader)
            for batch_id, (source_loader,
                           target_loader) in enumerate(zip_loader):
                self.before_training_step(batch_id, num_batches)
                self.training_step(source_loader, target_loader, batch_id,
                                   num_batches, epoch_id, num_epochs)
                self.after_training_step(batch_id, num_batches)
            self.after_training_epoch(epoch_id + 1, num_epochs)

            self.before_validation_epoch(epoch_id + 1, num_epochs)
            for batch_id, val_batch in enumerate(val_loader):
                num_batches = len(val_loader)
                self.before_validation_step(batch_id, num_batches)
                self.validation_step(val_batch, batch_id, num_batches)
                self.after_validation_step(batch_id, num_batches)
            self.after_validation_epoch(epoch_id + 1, num_epochs)
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
            'domain_classifier': nn.Module,
            'optimizer': torch.optim.Optimizer,
            'loss_fn': nn.Module,
            'device': torch.device,
            'lambd': float
        }


class DANNTrainer(DANNTrainerInterface):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Domain Adversarial Neural Networks (DANN) for deep domain adaptation.

    NOTE: DANN belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. The journal of machine learning research, 2016, 17(1): 2096-2030.
    - URL: https://arxiv.org/abs/1505.07818
    - Related Project: https://github.com/fungtion/DANN/blob/master/train/main.py

    .. code-block:: python

        trainer = DANNTrainer(extractor, classifier, domain_classifier)
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

        class DANNTrainer(DANNTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.extractor = self.extractor.to(self.device)
                self.classifier = self.classifier.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class DANNTrainer(DANNTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                super().before_training_epoch(epoch_id, num_epochs)
    
    Args:
        extractor (nn.Module): The feature extraction model, learning the feature representation of EEG signal by forcing the correlation matrixes of source and target data close.
        classifier (nn.Module): The classification model, learning the classification task with source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        domain_classifier (nn.Module): The classification model, learning to discriminate between the source and target domains based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function or a gradient reverse layer (which is already included in the implementation).
        lambd (float): The weight of DANN loss to trade-off between the classification loss and DANN loss. (defualt: :obj:`1.0`)
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        device: (torch.device or str): The device on which the model and data is or will be allocated. (defualt: :obj:`False`)
    
    .. automethod:: fit
    .. automethod:: score
    '''
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 domain_classifier: nn.Module,
                 lambd: float = 1.0,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device=torch.device('cpu')):
        super(DANNTrainer, self).__init__()

        self.extractor = extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        self.lambd = lambd

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
