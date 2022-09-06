from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from .utils import AverageMeter


class BaseTrainerInterface:
    r'''
    A generic trainer interface for EEG classification. To use this interface, at least :obj:`model`, :obj:`optimizer`, :obj:`loss_fn`, :obj:`device` needs to be initialized in :obj:`__init__`. If you don't need to customize your own trainer, you can simply use :obj:`BaseTrainer`.

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

        class BaseTrainer(BaseTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.model = self.model.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class BaseTrainer(BaseTrainerInterface):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                super().before_training_epoch(epoch_id, num_epochs)
        
    '''
    def __init__(self):
        self.val_loss = AverageMeter()
        self.val_correct = AverageMeter()

    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        print(f"Epoch {epoch_id}\n-------------------------------")
        self.model = self.model.to(self.device)

    def before_training_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def training_step(self, train_batch: Tuple, batch_id: int,
                      num_batches: int):
        self.model.train()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if batch_id % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_id:>5d}/{num_batches:>5d}]")

    def after_training_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def after_training_epoch(self, epoch_id: int, num_epochs: int):
        # optional hooks
        pass

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.model = self.model.to(self.device)
        self.val_loss.reset()
        self.val_correct.reset()

    def before_validation_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def validation_step(self, val_batch: Tuple, batch_id: int,
                        num_batches: int):
        self.model.eval()
        with torch.no_grad():
            X = val_batch[0].to(self.device)
            y = val_batch[1].to(self.device)

            pred = self.model(X)

            self.val_loss.update(self.loss_fn(pred, y).item())
            self.val_correct.update((pred.argmax(1) == y).float().sum().item(),
                                    n=X.shape[0])

    def after_validation_step(self, batch_id: int, num_batches: int):
        # optional hooks
        pass

    def after_validation_epoch(self, epoch_id: int, num_epochs: int):
        print(
            f"\nTest Error: Avg accuracy: {(100 * self.val_correct.avg):>0.1f}%, Avg loss: {self.val_loss.avg:>8f}"
        )

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        self._check_dependencies()
        for t in range(num_epochs):
            self.before_training_epoch(t + 1, num_epochs)
            num_batches = len(train_loader)
            for batch_id, train_batch in enumerate(train_loader):
                self.before_training_step(batch_id, num_batches)
                self.training_step(train_batch, batch_id, num_batches)
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

    def score(self, test_loader: DataLoader):
        self._check_dependencies()

        self.model.eval()
        correct = AverageMeter()

        with torch.no_grad():
            for batch_id, val_batch in enumerate(test_loader):
                X = val_batch[0].to(self.device)
                y = val_batch[1].to(self.device)
                pred = self.model(X)
                correct.update((pred.argmax(1) == y).float().sum().item(),
                               X.shape[0])
        return correct.avg

    def _dependencies(self):
        return {
            'model': nn.Module,
            'optimizer': torch.optim.Optimizer,
            'loss_fn': nn.Module,
            'device': torch.device
        }

    def _check_dependencies(self):
        for dependency, dependency_type in self._dependencies().items():
            assert hasattr(
                self, dependency
            ), f'The {dependency} needs to be defined in the initialization function!'
            assert isinstance(
                getattr(self, dependency), dependency_type
            ), f'The data type of {dependency} should be {dependency_type}, but got {type(getattr(self, dependency))}!'


class BaseTrainer(BaseTrainerInterface):
    r'''
    A generic trainer class for EEG classification.

    .. code-block:: python

        trainer = BaseTrainer(model)
        trainer.fit(train_loader, val_loader)
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

        class MyBaseTrainer(BaseTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                print(f"Epoch {epoch_id}\n-------------------------------")
                self.model = self.model.to(self.device)

    If you don't want to change the original implementation while supplementing new features, you can consider calling the interface method after the customized implementation:

    .. code-block:: python

        class MyBaseTrainer(BaseTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                print("Do something here.")
                super().before_training_epoch(epoch_id, num_epochs)
    
    Args:
        model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        device: (torch.device or str): The device on which the model and data is or will be allocated. (defualt: :obj:`False`)
    
    .. automethod:: fit
    .. automethod:: score
    '''
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device: Union[torch.device, str]=torch.device('cpu')):
        super(BaseTrainer, self).__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        return super().fit(train_loader=train_loader,
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