import math
from typing import List, Tuple, Optional

import torch
import numpy as np
import torch.nn as nn
from .utils import classification_metrics
import torchmetrics
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class ClassificationTrainer(BasicTrainer):
    r'''
    A generic trainer class for EEG classification.

    .. code-block:: python

        trainer = ClassificationTrainer(model)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    The class provides the following hook functions for inserting additional implementations in the training, validation and testing lifecycle:

    - :obj:`before_training_epoch`: executed before each epoch of training starts
    - :obj:`before_training_step`: executed before each batch of training starts
    - :obj:`on_training_step`: the training process for each batch
    - :obj:`after_training_step`: execute after the training of each batch
    - :obj:`after_training_epoch`: executed after each epoch of training
    - :obj:`before_validation_epoch`: executed before each round of validation starts
    - :obj:`before_validation_step`: executed before the validation of each batch
    - :obj:`on_validation_step`: validation process for each batch
    - :obj:`after_validation_step`: executed after the validation of each batch
    - :obj:`after_validation_epoch`: executed after each round of validation
    - :obj:`before_test_epoch`: executed before each round of test starts
    - :obj:`before_test_step`: executed before the test of each batch
    - :obj:`on_test_step`: test process for each batch
    - :obj:`after_test_step`: executed after the test of each batch
    - :obj:`after_test_epoch`: executed after each round of test

    If you want to customize some operations, you just need to inherit the class and override the hook function:

    .. code-block:: python

        class MyClassificationTrainer(ClassificationTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = ClassificationTrainer(model, device_ids=[1, 2, 7])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Then, you can use the :obj:`torch.distributed.launch` or :obj:`torchrun` to run your python file.

    .. code-block:: shell

        python -m torch.distributed.launch \
            --nproc_per_node=3 \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr="localhost" \
            --master_port=2345 \
            your_python_file.py

    Here, :obj:`nproc_per_node` is the number of GPUs you specify.

    Args:
        model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
        metrics (list, optional): Supported metrics are precision, recall, f1_score and accuracy. atleast one should be present
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        device_ids (list): Use cpu if the list is empty. If the list contains indices of multiple GPUs, it needs to be launched with :obj:`torch.distributed.launch` or :obj:`torchrun`. (defualt: :obj:`[]`)
        ddp_sync_bn (bool): Whether to replace batch normalization in network structure with cross-GPU synchronized batch normalization. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_replace_sampler (bool): Whether to replace sampler in dataloader with :obj:`DistributedSampler`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_val (bool): Whether to use multi-GPU acceleration for the validation set. For experiments where data input order is sensitive, :obj:`ddp_val` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_test (bool): Whether to use multi-GPU acceleration for the test set. For experiments where data input order is sensitive, :obj:`ddp_test` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
    
    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 model: nn.Module,
                 num_classes: Optional[int] = None,
                 metrics: Optional[list] = None, 
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(ClassificationTrainer,
              self).__init__(modules={'model': model},
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics = metrics
        
        if not num_classes is None:
            self.num_classes = num_classes
        elif hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        else:
            raise ValueError('The number of classes is not specified.')

        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        # init metric
        self.train_loss = torchmetrics.MeanMetric().to(self.device)

        self.train_metrics = classification_metrics(metric_list=self.metrics,
                                                    num_classes=self.num_classes,
                                                    device=self.device)

        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_metrics = classification_metrics(metric_list=self.metrics,
                                                  num_classes=self.num_classes,
                                                  device=self.device)

        self.test_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_metrics = classification_metrics(metric_list=self.metrics,
                                                   num_classes=self.num_classes,
                                                   device=self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_metrics.reset()
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        # compute prediction error
        pred = self.modules['model'](X)
        loss = self.loss_fn(pred, y)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)
            train_loss = self.train_loss.compute()

            # Update and compute selected metrics
            
            self.train_metrics.update(pred.argmax(1), y)
            train_metric_results = self.train_metrics.compute()
            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size

            log_msg = f"loss: {train_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
            for metric, result in train_metric_results.items():
                log_msg += f" {metric.replace('Multiclass','')}: {result*100:>0.3f}"
            self.log(log_msg)


    def before_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.val_metrics.reset()
        self.val_loss.reset()

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        pred = self.modules['model'](X)

        self.val_loss.update(self.loss_fn(pred, y))
        self.val_metrics.update(pred.argmax(1), y)
        
    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        val_loss = self.val_loss.compute()
        val_metric_results = self.val_metrics.compute()
        log_msg = f"\nloss: {val_loss:>8f} "
        for metric, result in val_metric_results.items():
            log_msg += f" {metric.replace('Multiclass','')}: {result*100:>0.3f} "
        self.log(log_msg)

    def before_test_epoch(self, **kwargs):
        self.test_metrics.reset()
        self.test_loss.reset()

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        pred = self.modules['model'](X)

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_metrics.update(pred.argmax(1), y)

    def after_test_epoch(self, **kwargs):
        test_metric_results = self.test_metrics.compute()
        test_loss = self.test_loss.compute()
        log_msg = f"\nloss: {test_loss:>8f} "
        for metric, result in test_metric_results.items():
            log_msg += f" {metric.replace('Multiclass','')}: {result*100:>0.3f}"
        self.log(log_msg)

    def test(self, test_loader: DataLoader, **kwargs):
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        super().test(test_loader=test_loader, **kwargs)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1,
            **kwargs):
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        super().fit(train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    **kwargs)
