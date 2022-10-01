import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class GANTrainer(BasicTrainer):
    r'''
    A generic trainer class for EEG classification.

    .. code-block:: python

        trainer = GANTrainer(generator, discriminator)
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

        class MyGANTrainer(GANTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = GANTrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        generator (nn.Module): The generator model for EEG signal generation, whose inputs are Gaussian distributed random vectors, outputs are generated EEG signals. The dimensions of the input vector should be defined on the :obj:`in_channel` attribute. The output layer does not need to have a softmax activation function.
        discriminator (nn.Module): The discriminator model to determine whether the EEG signal is real or generated, and the dimension of its output should be equal to the one (i.e., the score to distinguish the real and the fake). The output layer does not need to have a sigmoid activation function.
        generator_lr (float): The learning rate of the generator. (defualt: :obj:`0.0001`)
        discriminator_lr (float): The learning rate of the discriminator. (defualt: :obj:`0.0001`)
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
                 generator: nn.Module,
                 discriminator: nn.Module,
                 generator_lr: float = 1e-4,
                 discriminator_lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(GANTrainer,
              self).__init__(modules={
                  'generator': generator,
                  'discriminator': discriminator,
              },
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.weight_decay = weight_decay

        self.generator_optimizer = torch.optim.Adam(generator.parameters(),
                                                    lr=generator_lr,
                                                    weight_decay=weight_decay)
        self.discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=discriminator_lr,
            weight_decay=weight_decay)

        self.loss_fn = nn.BCEWithLogitsLoss()
        # init metric
        self.train_g_loss = torchmetrics.MeanMetric().to(self.device)
        self.train_d_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_g_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_d_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_g_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_d_loss = torchmetrics.MeanMetric().to(self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int):
        self.train_g_loss.reset()
        self.train_d_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        # backpropagation for generator
        valid = torch.ones((X.shape[0], 1), device=self.device)
        fake = torch.zeros((X.shape[0], 1), device=self.device)

        self.generator_optimizer.zero_grad()

        assert hasattr(
            self.modules['generator'], 'in_channels'
        ), 'The generator must have the property in_channels to generate a batch of latent codes for the corresponding dimension.'

        z = torch.normal(mean=0,
                         std=1,
                         size=(X.shape[0],
                               self.modules['generator'].in_channels)).to(self.device)
        gen_X = self.modules['generator'](z)
        g_loss = self.loss_fn(self.modules['discriminator'](gen_X), valid)

        g_loss.backward()
        self.generator_optimizer.step()

        # backpropagation for discriminator
        self.discriminator_optimizer.zero_grad()

        real_loss = self.loss_fn(self.modules['discriminator'](X), valid)
        fake_loss = self.loss_fn(self.modules['discriminator'](gen_X.detach()),
                                 fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.discriminator_optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_g_loss.update(g_loss)
            self.train_d_loss.update(d_loss)

            train_g_loss = self.train_g_loss.compute()
            train_d_loss = self.train_d_loss.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"g_loss: {train_g_loss:>8f}, d_loss: {train_d_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.val_g_loss.reset()
        self.val_d_loss.reset()

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        valid = torch.ones((X.shape[0], 1), device=self.device)
        fake = torch.zeros((X.shape[0], 1), device=self.device)

        # for g_loss
        z = torch.normal(mean=0,
                         std=1,
                         size=(X.shape[0],
                               self.modules['generator'].in_channels)).to(self.device)
        gen_X = self.modules['generator'](z)
        g_loss = self.loss_fn(self.modules['discriminator'](gen_X), valid)

        # for d_loss
        real_loss = self.loss_fn(self.modules['discriminator'](X), valid)
        fake_loss = self.loss_fn(self.modules['discriminator'](gen_X.detach()),
                                 fake)
        d_loss = (real_loss + fake_loss) / 2

        self.val_g_loss.update(g_loss)
        self.val_d_loss.update(d_loss)

    def after_validation_epoch(self, epoch_id: int, num_epochs: int):
        val_g_loss = self.val_g_loss.compute()
        val_d_loss = self.val_d_loss.compute()
        self.log(f"\ng_loss: {val_g_loss:>8f}, d_loss: {val_d_loss:>8f}")

    def before_test_epoch(self):
        self.test_g_loss.reset()
        self.test_d_loss.reset()

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        valid = torch.ones((X.shape[0], 1), device=self.device)
        fake = torch.zeros((X.shape[0], 1), device=self.device)

        # for g_loss
        z = torch.normal(mean=0,
                         std=1,
                         size=(X.shape[0],
                               self.modules['generator'].in_channels)).to(self.device)
        gen_X = self.modules['generator'](z)
        g_loss = self.loss_fn(self.modules['discriminator'](gen_X), valid)

        # for d_loss
        real_loss = self.loss_fn(self.modules['discriminator'](X), valid)
        fake_loss = self.loss_fn(self.modules['discriminator'](gen_X.detach()),
                                 fake)
        d_loss = (real_loss + fake_loss) / 2

        self.test_g_loss.update(g_loss)
        self.test_d_loss.update(d_loss)

    def after_test_epoch(self):
        test_g_loss = self.test_g_loss.compute()
        test_d_loss = self.test_d_loss.compute()
        self.log(f"\ng_loss: {test_g_loss:>8f}, d_loss: {test_d_loss:>8f}")

    def test(self, test_loader: DataLoader):
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        super().test(test_loader=test_loader)

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
        super().fit(train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs)
