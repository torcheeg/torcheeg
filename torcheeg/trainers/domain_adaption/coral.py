import math
from itertools import chain, cycle
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data.dataloader import DataLoader

from ..classification_trainer import ClassificationTrainer


class CORALTrainer(ClassificationTrainer):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of CORrelation ALignment (CORAL) for deep domain adaptation.

    NOTE: CORAL belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Sun B, Saenko K. Deep CORAL: Correlation alignment for deep domain adaptation[C]//European conference on computer vision. Springer, Cham, 2016: 443-450.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/adapt-python/adapt/blob/master/adapt/feature_based/_deepCORAL.py

    .. code-block:: python

        trainer = CORALTrainer(extractor, classifier)
        trainer.fit(source_loader, target_loader, val_loader)
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

        class MyCORALTrainer(CORALTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)

    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = CORALTrainer(model, device_ids=[1, 2, 7])
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
        extractor (nn.Module): The feature extraction model, learning the feature representation of EEG signal by forcing the correlation matrixes of source and target data close.
        classifier (nn.Module): The classification model, learning the classification task with source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        lambd (float): The weight of CORAL loss to trade-off between the classification loss and CORAL loss. (defualt: :obj:`1.0`)
        match_mean (bool): Weither to match the means of the source domain and target domain samples. If :obj:`False`, only the second moment is matched. (defualt: :obj:`False`)
        num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        device_ids (list): Use cpu if the list is empty. If the list contains indices of multiple GPUs, it needs to be launched with :obj:`torch.distributed.launch` or :obj:`torchrun`. (defualt: :obj:`[]`)
        ddp_sync_bn (bool): Whether to replace batch normalization in network structure with cross-GPU synchronized batch normalization. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_replace_sampler (bool): Whether to replace sampler in dataloader with :obj:`DistributedSampler`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_val (bool): Whether to use multi-GPU acceleration for the validation set. For experiments where data input order is sensitive, :obj:`ddp_val` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_test (bool): Whether to use multi-GPU acceleration for the test set. For experiments where data input order is sensitive, :obj:`ddp_test` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
    
    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 match_mean: bool = True,
                 lambd: float = 1.0,
                 num_classes: Optional[int] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        # call BasicTrainer
        # pylint: disable=bad-super-call
        super(ClassificationTrainer,
              self).__init__(modules={
                  'extractor': extractor,
                  'classifier': classifier
              },
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.weight_decay = weight_decay
        self.match_mean = match_mean
        self.lambd = lambd

        if not num_classes is None:
            self.num_classes = num_classes
        elif hasattr(classifier, 'num_classes'):
            self.num_classes = classifier.num_classes
        else:
            raise ValueError('The number of classes is not specified.')

        self.optimizer = torch.optim.Adam(chain(extractor.parameters(),
                                                classifier.parameters()),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        # init metric
        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(self.device)

        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(self.device)

        self.test_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(self.device)

    def on_training_step(self, source_loader: DataLoader,
                         target_loader: DataLoader, batch_id: int,
                         num_batches: int, epoch_id: int, num_epochs: int,
                         **kwargs):
        self.train_accuracy.reset()
        self.train_loss.reset()

        if self.match_mean:
            match_mean = 1.0
        else:
            match_mean = 0.0

        self.optimizer.zero_grad()

        X_source = source_loader[0].to(self.device)
        y_source = source_loader[1].to(self.device)
        X_target = target_loader[0].to(self.device)

        X_source_feat = self.modules['extractor'](X_source)
        y_source_pred = self.modules['classifier'](X_source_feat)
        X_target_feat = self.modules['extractor'](X_target)

        batch_size = X_source.shape[0]

        # Compute the loss value
        batch_size = X_source_feat.shape[1]

        # source covariance
        X_source_devi = torch.mean(X_source_feat, 1,
                                   keepdim=True) - X_source_feat
        X_source_cova = torch.matmul(torch.transpose(X_source_devi, 0, 1),
                                     X_source_devi)

        # target covariance
        X_target_devi = torch.mean(X_target_feat, 1,
                                   keepdim=True) - X_target_feat
        X_target_cova = torch.matmul(torch.transpose(X_target_devi, 0, 1),
                                     X_target_devi)
        # frobenius norm between source and target
        coral_loss = torch.mean(
            torch.mul((X_source_cova - X_target_cova),
                      (X_source_cova - X_target_cova))) / (4 * batch_size * 4)

        p = float(batch_id + epoch_id * num_batches) / num_epochs / num_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        coral_loss = self.lambd * alpha * coral_loss
        task_loss = self.loss_fn(y_source_pred, y_source)

        loss = task_loss + coral_loss

        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)
            self.train_accuracy.update(y_source_pred.argmax(1), y_source)

            train_loss = self.train_loss.compute()
            train_accuracy = 100 * self.train_accuracy.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"loss: {train_loss:>8f}, accuracy: {train_accuracy:>0.1f}% [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def fit(self,
            source_loader: DataLoader,
            target_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1,
            **kwargs):
        r'''
        Args:
            source_loader (DataLoader): Iterable DataLoader for traversing the data batch from the source domain (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            target_loader (DataLoader): Iterable DataLoader for traversing the training data batch from the target domain (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc). The target dataset does not have to return labels.
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        source_loader = self.on_reveive_dataloader(source_loader, mode='train')
        target_loader = self.on_reveive_dataloader(target_loader, mode='train')
        val_loader = self.on_reveive_dataloader(val_loader, mode='val')

        for t in range(num_epochs):
            if hasattr(source_loader, 'need_to_set_epoch'):
                source_loader.sampler.set_epoch(t)
            if hasattr(target_loader, 'need_to_set_epoch'):
                target_loader.sampler.set_epoch(t)
            if hasattr(val_loader, 'need_to_set_epoch'):
                val_loader.sampler.set_epoch(t)

            num_batches = max(len(source_loader), len(target_loader))

            # set model to train mode
            for k, m in self.modules.items():
                self.modules[k].train()

            zip_loader = zip(source_loader, cycle(target_loader)) if len(
                source_loader) > len(target_loader) else zip(
                    cycle(source_loader), target_loader)

            # hook
            self.before_training_epoch(t + 1, num_epochs, **kwargs)
            for batch_id, (cur_source_loader,
                           cur_target_loader) in enumerate(zip_loader):
                # hook
                self.before_training_step(batch_id, num_batches, **kwargs)
                # hook
                self.on_training_step(cur_source_loader, cur_target_loader,
                                      batch_id, num_batches, t, num_epochs,
                                      **kwargs)
                # hook
                self.after_training_step(batch_id, num_batches, **kwargs)
            # hook
            self.after_training_epoch(t + 1, num_epochs, **kwargs)

            # set model to val mode
            for k, m in self.modules.items():
                self.modules[k].eval()

            num_batches = len(val_loader)
            # hook
            self.before_validation_epoch(t + 1, num_epochs, **kwargs)
            for batch_id, val_batch in enumerate(val_loader):
                # hook
                self.before_validation_step(batch_id, num_batches, **kwargs)
                # hook
                self.on_validation_step(val_batch, batch_id, num_batches,
                                        **kwargs)
                # hook
                self.after_validation_step(batch_id, num_batches, **kwargs)
                # hook
            self.after_validation_epoch(t + 1, num_epochs, **kwargs)
        return self

    def test(self, test_loader: DataLoader, **kwargs):
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        super().test(test_loader=test_loader, **kwargs)

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        feat = self.modules['extractor'](X)
        pred = self.modules['classifier'](feat)

        self.val_loss.update(self.loss_fn(pred, y))
        self.val_accuracy.update(pred.argmax(1), y)

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        feat = self.modules['extractor'](X)
        pred = self.modules['classifier'](feat)

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)
