import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from itertools import chain
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class DDPMTrainer(BasicTrainer):
    r'''
    The diffusion model consists of two processes, the forward process, and the backward process. The forward process is to gradually add Gaussian noise to an image until it becomes random noise, while the backward process is the de-noising process. We train an attention-based UNet network at the backward process to start with random noise and gradually de-noise it until an image is generated and use the UNet to generate a simulated image from random noises. This class implements the training, test, and new sample inference of DDPM.

    .. code-block:: python

        unet = BUNet(in_channels=4)
        trainer = DDPMTrainer(unet)
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

        class MyDDPMTrainer(DDPMTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = DDPMTrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        beta_start: (float): The start point of the linear beta scheduler to sample betas. (defualt: :obj:`1e-4`)
        beta_end: (float): The end point of the linear beta scheduler to sample betas. (defualt: :obj:`2e-2`)
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
                 unet: nn.Module,
                 lr: float = 3e-4,
                 beta_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(DDPMTrainer,
              self).__init__(modules={
                  'unet': unet,
              },
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.beta_timesteps = beta_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.linear_beta_schedule(self.beta_timesteps,
                                              self.beta_start, self.beta_end)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()
        # init metric
        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_loss = torchmetrics.MeanMetric().to(self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        # prepare noise schedule
        return torch.linspace(start, end, timesteps).to(self.device)

    def forward_diffusion_samples(self, x_0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None,
                                                                     None, None]
        noise = torch.randn_like(x_0)
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, num_samples):
        return torch.randint(low=1,
                             high=self.beta_timesteps,
                             size=(num_samples, )).to(self.device)

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t)
        loss = self.loss_fn(noise, noise_pred)

        loss.backward()
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

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t)
        loss = self.loss_fn(noise, noise_pred)

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

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t)
        loss = self.loss_fn(noise, noise_pred)

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

    def sample(self, num_samples: int, sample_size: Tuple[int]) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.
            sample_size (tuple): Shape of a sample.

        Returns:
            torch.Tensor: the generated samples.
        """
        self.modules['unet'].eval()
        with torch.no_grad():
            samples = torch.randn((num_samples, *sample_size)).to(self.device)
            for i in reversed(range(1, self.beta_timesteps)):
                t = (torch.ones(
                    num_samples, dtype=torch.long, device=self.device) * i)
                noise_pred = self.modules['unet'](samples, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(samples)
                else:
                    noise = torch.zeros_like(samples)

                samples = 1 / torch.sqrt(alpha) * (samples - (
                    (1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred
                                                   ) + torch.sqrt(beta) * noise
            return samples


class CDDPMTrainer(DDPMTrainer):
    r'''
    The diffusion model consists of two processes, the forward process, and the backward process. The forward process is to gradually add Gaussian noise to an image until it becomes random noise, while the backward process is the de-noising process. We train an attention-based UNet network at the backward process to start with random noise and gradually de-noise it until an image is generated and use the UNet to generate a simulated image from random noises. In particular, in conditional UNet, additional label information is provided to guide the noise reduction results during the noise reduction process. This class implements the training, test, and new sample inference of the conditional DDPM.

    .. code-block:: python

        unet = BCUNet(in_channels=4, num_classes=2)
        trainer = CDDPMTrainer(unet)
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

        class MyCDDPMTrainer(CDDPMTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = CDDPMTrainer(generator, discriminator, device_ids=[1, 2, 7])
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
        beta_start: (float): The start point of the linear beta scheduler to sample betas. (defualt: :obj:`1e-4`)
        beta_end: (float): The end point of the linear beta scheduler to sample betas. (defualt: :obj:`2e-2`)
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
                 unet: nn.Module,
                 lr: float = 3e-4,
                 beta_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(CDDPMTrainer, self).__init__(
            unet=unet,
            lr=lr,
            beta_timesteps=beta_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device_ids=device_ids,
            ddp_sync_bn=ddp_sync_bn,
            ddp_replace_sampler=ddp_replace_sampler,
            ddp_val=ddp_val,
            ddp_test=ddp_test,
        )

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int, **kwargs):
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t, y)
        loss = self.loss_fn(noise, noise_pred)

        loss.backward()
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

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t, y)
        loss = self.loss_fn(noise, noise_pred)

        self.val_loss.update(loss)

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.forward_diffusion_samples(X, t)
        noise_pred = self.modules['unet'](x_t, t, y)
        loss = self.loss_fn(noise, noise_pred)

        self.test_loss.update(loss)

    def sample(self,
               num_samples: int,
               sample_size: Tuple[int],
               labels: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.
            sample_size (tuple): Shape of a sample.
            labels (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size. If not provided, a batch of randomly generated categories will be used.

        Returns:
            torch.Tensor: the generated samples.
        """
        if not labels is None:
            assert len(
                labels
            ) == num_samples, f'labels ({len(labels)}) should be the same length as num_samples ({num_samples}).'
            assert isinstance(
                labels, torch.Tensor
            ), f'labels should be torch.Tensor instances, the current input is {type(labels)}'
        else:
            labels = torch.randint(low=0,
                                   high=self.modules['unet'].num_classes,
                                   size=(num_samples, ))
        labels = labels.long().to(self.device)
        self.modules['unet'].eval()
        with torch.no_grad():
            samples = torch.randn((num_samples, *sample_size)).to(self.device)
            for i in reversed(range(1, self.beta_timesteps)):
                t = (torch.ones(
                    num_samples, dtype=torch.long, device=self.device) * i)
                noise_pred = self.modules['unet'](samples, t, labels)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(samples)
                else:
                    noise = torch.zeros_like(samples)

                samples = 1 / torch.sqrt(alpha) * (samples - (
                    (1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred
                                                   ) + torch.sqrt(beta) * noise
            return samples