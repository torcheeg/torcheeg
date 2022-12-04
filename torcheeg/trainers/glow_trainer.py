import math
from itertools import chain
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class GlowTrainer(BasicTrainer):
    r'''
    This class implement the training, test, and new EEG inference of normalizing flow-based models. Glow is dedicated to train an encoder that encodes the input as a hidden variable and makes the hidden variable obey the standard normal distribution. By good design, the encoder should be reversible. On this basis, as soon as the encoder is trained, the corresponding decoder can be used to generate samples from a Gaussian distribution according to the inverse operation. In particular, compared with vanilla normalizing flow-based models, the Glow model is a easy-to-use flow-based model that replaces the operation of permutating the channel axes by introducing a 1x1 reversible convolution.
    
    Below is a recommended suite for use in EEG generation:

    .. code-block:: python

        model = BGlow(in_channels=4)
        trainer = GlowTrainer(generator, discriminator)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Below is a recommended suite for use in conditional EEG generation:

    .. code-block:: python

        model = BGlow(in_channels=4, num_classes=2)
        trainer = GlowTrainer(generator, discriminator)
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

        class MyGlowTrainer(GlowTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = GlowTrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        encoder (nn.Module): The encoder, whose inputs are EEG signals, outputs are two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick.
        decoder (nn.Module): The decoder generating EEG signals from hidden variables encoded by the encoder. The dimensions of the input vector should be defined on the :obj:`in_channel` attribute.
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        beta: (float): The weight of the KL divergence in the loss function. Please refer to betaGlow. (defualt: :obj:`1.0`)
        device_ids (list): Use cpu if the list is empty. If the list contains indices of multiple GPUs, it needs to be launched with :obj:`torch.distributed.launch` or :obj:`torchrun`. (defualt: :obj:`[]`)
        ddp_sync_bn (bool): Whether to replace batch normalization in network structure with cross-GPU synchronized batch normalization. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_replace_sampler (bool): Whether to replace sampler in dataloader with :obj:`DistributedSampler`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_val (bool): Whether to use multi-GPU acceleration for the validation set. For experiments where data input order is sensitive, :obj:`ddp_val` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_test (bool): Whether to use multi-GPU acceleration for the test set. For experiments where data input order is sensitive, :obj:`ddp_test` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
    
    .. automethod:: fit
    .. automethod:: test
    .. automethod:: sample
    '''
    def __init__(self,
                 glow: nn.Module,
                 lr: float = 1e-4,
                 grad_norm_clip: float = 50.0,
                 loss_scale: float = 1e-3,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(GlowTrainer,
              self).__init__(modules={'glow': glow},
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.loss_scale = loss_scale
        self.grad_norm_clip = grad_norm_clip
        self.optimizer = torch.optim.Adam(glow.parameters(), lr=lr)

        # init metric
        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_loss = torchmetrics.MeanMetric().to(self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def log_prob(self, value, loc=0.0, scale=1.0):
        var = (scale**2)
        log_scale = math.log(scale)
        return -((value - loc)**2) / (2 * var) - log_scale - math.log(
            math.sqrt(2 * math.pi))

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        if self.modules['glow'].num_classes > 0:
            _, nll_loss, y_pred = self.modules['glow'](X, y)
            cls_loss = F.cross_entropy(y_pred, y)
            loss = nll_loss.mean() + cls_loss
        else:
            _, nll_loss, _ = self.modules['glow'](X)
            loss = nll_loss.mean()

        try:
            loss.backward()  # svd_cuda: For ... is zero, singular U.
        except Exception as e:
            self.log('[WARNING] svd_cuda: Find zero items, singular U.')
            return

        nn.utils.clip_grad_norm_(self.modules['glow'].parameters(),
                                 self.grad_norm_clip)

        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)

            train_loss = self.train_loss.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"loss: {train_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def before_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.val_loss.reset()

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        if self.modules['glow'].num_classes > 0:
            _, nll_loss, y_pred = self.modules['glow'](X, y)
            cls_loss = F.cross_entropy(y_pred, y)
            loss = nll_loss.mean() + cls_loss
        else:
            _, nll_loss, _ = self.modules['glow'](X)
            loss = nll_loss.mean()

        self.val_loss.update(loss)

    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        val_loss = self.val_loss.compute()
        self.log(f"\nloss: {val_loss:>8f}")

    def before_test_epoch(self, **kwargs):
        self.test_loss.reset()

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        if self.modules['glow'].num_classes > 0:
            _, nll_loss, y_pred = self.modules['glow'](X, y)
            cls_loss = F.cross_entropy(y_pred, y)
            loss = nll_loss.mean() + cls_loss
        else:
            _, nll_loss, _ = self.modules['glow'](X)
            loss = nll_loss.mean()

        self.test_loss.update(loss)

    def after_test_epoch(self, **kwargs):
        test_loss = self.test_loss.compute()
        self.log(f"\nloss: {test_loss:>8f}")

    def test(self, test_loader: DataLoader, **kwargs):
        r'''
        Validate the performance of the model on the test set.

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
        Train the model on the training set and use the validation set to validate the results of each round of training.

        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        super().fit(train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    **kwargs)

    def sample(self,
               num_samples: int,
               temperature: float = 1.0,
               labels: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.
            temperature (float): The hyper-parameter, temperature, to sample from gaussian distributions. (defualt: :obj:`1.0`)
            labels (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size. If not provided, a batch of randomly generated categories will be used.

        Returns:
            torch.Tensor: the generated samples.
        """
        if labels:
            assert len(
                labels
            ) == num_samples, f'labels ({len(labels)}) should be the same length as num_samples ({num_samples}).'
            assert isinstance(
                labels, torch.Tensor
            ), f'labels should be torch.Tensor instances, the current input is {type(labels)}'
        else:
            labels = torch.randint(low=0,
                                   high=self.modules['glow'].num_classes,
                                   size=(num_samples, ))
        labels = labels.long().to(self.device)
        self.modules['glow'].eval()
        with torch.no_grad():
            return self.modules['glow'](labels=labels, temperature=temperature, reverse=True)
