import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from itertools import chain
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class VAETrainer(BasicTrainer):
    r'''
    The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the training, test, and new EEG inference of variational autoencoders.

    .. code-block:: python

        encoder = BEncoder(in_channels=4)
        decoder = BDecoder(in_channels=64, out_channels=4)
        trainer = VAETrainer(encoder, decoder)
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

        class MyVAETrainer(VAETrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = VAETrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        beta: (float): The weight of the KL divergence in the loss function. Please refer to betaVAE. (defualt: :obj:`1.0`)
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
                 encoder: nn.Module,
                 decoder: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 beta: float = 1.0,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(VAETrainer,
              self).__init__(modules={
                  'encoder': encoder,
                  'decoder': decoder,
              },
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta

        self.optimizer = torch.optim.Adam(chain(encoder.parameters(),
                                                decoder.parameters()),
                                          lr=lr,
                                          weight_decay=weight_decay)

        self.loss_fn = nn.MSELoss()
        # init metric
        self.train_rec_loss = torchmetrics.MeanMetric().to(self.device)
        self.train_kld_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_rec_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_kld_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_rec_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_kld_loss = torchmetrics.MeanMetric().to(self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_rec_loss.reset()
        self.train_kld_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        try:
            mu, log_var = self.modules['encoder'](X)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )

        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        # Bias correction term accounting for the minibatch samples from the dataset. It is necessary when small batch-sizes are used, which can lead to a large variance in the KLD value. Some other studies use latent dimension/the input dimension instead.
        loss = rec_loss + self.beta * (1 / num_batches) * kld_loss

        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_rec_loss.update(rec_loss)
            self.train_kld_loss.update(kld_loss)

            train_rec_loss = self.train_rec_loss.compute()
            train_kld_loss = self.train_kld_loss.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"rec_loss: {train_rec_loss:>8f}, kld_loss: {train_kld_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def before_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.val_rec_loss.reset()
        self.val_kld_loss.reset()

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        try:
            mu, log_var = self.modules['encoder'](X)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )
        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        self.val_rec_loss.update(rec_loss)
        self.val_kld_loss.update(kld_loss)

    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        val_rec_loss = self.val_rec_loss.compute()
        val_kld_loss = self.val_kld_loss.compute()
        self.log(
            f"\nrec_loss: {val_rec_loss:>8f}, kld_loss: {val_kld_loss:>8f}")

    def before_test_epoch(self, **kwargs):
        self.test_rec_loss.reset()
        self.test_kld_loss.reset()

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        try:
            mu, log_var = self.modules['encoder'](X)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )
        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        self.test_rec_loss.update(rec_loss)
        self.test_kld_loss.update(kld_loss)

    def after_test_epoch(self, **kwargs):
        test_rec_loss = self.test_rec_loss.compute()
        test_kld_loss = self.test_kld_loss.compute()
        self.log(
            f"\nrec_loss: {test_rec_loss:>8f}, kld_loss: {test_kld_loss:>8f}")

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

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: the generated samples.

        """
        self.modules['decoder'].eval()
        with torch.no_grad():
            z = torch.randn(num_samples,
                            self.modules['decoder'].in_channels).to(self.device)
            return self.modules['decoder'](z)


class CVAETrainer(VAETrainer):
    r'''
    The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the training, test, and new EEG inference of variational autoencoders.

    .. code-block:: python

        encoder = BCEncoder(in_channels=4, num_classes=2)
        decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=2)
        trainer = CVAETrainer(encoder, decoder)
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

        class MyCVAETrainer(CVAETrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = CVAETrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        beta: (float): The weight of the KL divergence in the loss function. Please refer to betaVAE. (defualt: :obj:`1.0`)
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
                 encoder: nn.Module,
                 decoder: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 beta: float = 1.0,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(CVAETrainer, self).__init__(
            encoder=encoder,
            decoder=decoder,
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            device_ids=device_ids,
            ddp_sync_bn=ddp_sync_bn,
            ddp_replace_sampler=ddp_replace_sampler,
            ddp_val=ddp_val,
            ddp_test=ddp_test,
        )

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_rec_loss.reset()
        self.train_kld_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        try:
            mu, log_var = self.modules['encoder'](X, y)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )

        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z, y)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        # Bias correction term accounting for the minibatch samples from the dataset. It is necessary when small batch-sizes are used, which can lead to a large variance in the KLD value. Some other studies use latent dimension/the input dimension instead.
        loss = rec_loss + self.beta * (1 / num_batches) * kld_loss

        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_rec_loss.update(rec_loss)
            self.train_kld_loss.update(kld_loss)

            train_rec_loss = self.train_rec_loss.compute()
            train_kld_loss = self.train_kld_loss.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"rec_loss: {train_rec_loss:>8f}, kld_loss: {train_kld_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        try:
            mu, log_var = self.modules['encoder'](X, y)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )
        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z, y)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        self.val_rec_loss.update(rec_loss)
        self.val_kld_loss.update(kld_loss)

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        try:
            mu, log_var = self.modules['encoder'](X, y)
        except:
            raise RuntimeError(
                'The output of the encoder should be two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick!'
            )
        z = self.reparameterize(mu, log_var)
        rec_X = self.modules['decoder'](z, y)

        rec_loss = self.loss_fn(rec_X, X)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        self.test_rec_loss.update(rec_loss)
        self.test_kld_loss.update(kld_loss)

    def sample(self,
               num_samples: int,
               labels: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.
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
                                   high=self.modules['decoder'].num_classes,
                                   size=(num_samples, ))
        labels = labels.long().to(self.device)
        self.modules['decoder'].eval()
        with torch.no_grad():
            z = torch.randn(num_samples,
                            self.modules['decoder'].in_channels).to(self.device)
            return self.modules['decoder'](z, labels)