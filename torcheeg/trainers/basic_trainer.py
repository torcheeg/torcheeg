import os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler


class BasicTrainer:
    r'''
    A generic trainer class for EEG analysis providing interfaces for all trainers to implement contexts common in training deep learning models. After inheriting this class, :obj:`on_training_step`, :obj:`on_validation_step`, and :obj:`on_test_step` must be implemented.

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

    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = BasicTrainer(model, device_ids=[1, 2, 7])
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
        model (Dict): A dictionary that stores neural networks for import, export and device conversion of neural networks.
        device_ids (list): Use cpu if the list is empty. If the list contains indices of multiple GPUs, it needs to be launched with :obj:`torch.distributed.launch` or :obj:`torchrun`. (defualt: :obj:`[]`)
        ddp_sync_bn (bool): Whether to replace batch normalization in network structure with cross-GPU synchronized batch normalization. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_replace_sampler (bool): Whether to replace sampler in dataloader with :obj:`DistributedSampler`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_val (bool): Whether to use multi-GPU acceleration for the validation set. For experiments where data input order is sensitive, :obj:`ddp_val` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_test (bool): Whether to use multi-GPU acceleration for the test set. For experiments where data input order is sensitive, :obj:`ddp_test` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
    
    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 modules: Dict,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        # given params
        self.modules = modules
        self.device_ids = device_ids
        self.ddp_sync_bn = ddp_sync_bn
        self.ddp_replace_sampler = ddp_replace_sampler
        self.ddp_val = ddp_val
        self.ddp_test = ddp_test

        # built-in params
        self.rank = -1
        self.local_rank = -1
        self.world_size = 1
        self.device = None

        # cpu trainer
        if len(device_ids) == 0:
            for k, m in self.modules.items():
                self.modules[k] = m.to(torch.device('cpu'))
            self.device = torch.device('cpu')

        # gpu trainer
        if len(device_ids) == 1:
            assert torch.cuda.is_available(
            ), 'GPU is not available, please set device to cpu!'
            for k, m in self.modules.items():
                self.modules[k] = m.to(torch.device('cuda', device_ids[0]))
            self.device = torch.device('cuda', device_ids[0])

        # ddp trainer
        if len(device_ids) > 1:
            assert torch.cuda.is_available(
            ), 'GPU is not available, please set device to cpu!'
            assert torch.cuda.device_count() > max(
                device_ids
            ), f'{torch.cuda.device_count()} GPUs are available, but try to access f{max(device_ids)}-th GPU!'

            rank = int(os.getenv('RANK', -1))
            local_rank = int(os.getenv('LOCAL_RANK', -1))
            world_size = int(os.getenv('WORLD_SIZE', 1))

            assert rank != -1, 'TorchEEG support to use multiple GPU with distributed data parallel, and you need to run the python script with torch.distributed.launch or torchrun.'

            self.rank = rank
            self.local_rank = local_rank
            self.world_size = world_size

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(d) for d in device_ids])

            assert dist.is_available(
            ), 'Distributed data parallel is not supported!'
            if not dist.is_initialized():
                # nccl is not available in windows
                backend = "nccl" if dist.is_nccl_available() else "gloo"
                if self.rank == 0:
                    self.log(f"Using backend: {backend}.")
                dist.init_process_group(backend=backend,
                                        rank=self.rank,
                                        world_size=self.world_size)

            for k, m in self.modules.items():
                self.modules[k] = m.to(torch.device('cuda', local_rank))

            for k, m in self.modules.items():
                ddp_m = DistributedDataParallel(m,
                                                device_ids=[self.local_rank],
                                                find_unused_parameters=True)
                if self.ddp_sync_bn:
                    ddp_m = nn.SyncBatchNorm.convert_sync_batchnorm(ddp_m)
                self.modules[k] = ddp_m

            self.device = torch.device(self.local_rank)

    @property
    def is_distributed(self):
        return self.rank != -1

    @property
    def is_main(self):
        # if not distributed mode -1, then it is the main process
        # if distributed mode > -1, then 0 is the main process
        return self.rank in {-1, 0}

    def log(self, *args, **kwargs):
        # can be overwritten
        if self.is_main:
            print(*args, **kwargs)

    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        # can be overwritten
        ...

    def before_training_step(self, batch_id: int, num_batches: int):
        # can be overwritten
        ...

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int):
        raise NotImplementedError

    def after_training_step(self, batch_id: int, num_batches: int):
        # can be overwritten
        ...

    def after_training_epoch(self, epoch_id: int, num_epochs: int):
        # can be overwritten
        ...

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        # can be overwritten
        ...

    def before_validation_step(self, batch_id: int, num_batches: int):
        # can be overwritten
        ...

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int):
        raise NotImplementedError

    def after_validation_step(self, batch_id: int, num_batches: int):
        # can be overwritten
        ...

    def after_validation_epoch(self, epoch_id: int, num_epochs: int):
        # can be overwritten
        ...

    def on_reveive_dataloader(self, dataloader, mode='train'):
        if mode == 'test' and not self.ddp_test:
            return dataloader
        if mode == 'val' and not self.ddp_val:
            return dataloader

        if not self.is_distributed:
            return dataloader

        shuffle = True
        if isinstance(dataloader.sampler, SequentialSampler):
            shuffle = False
        sampler = DistributedSampler(dataloader.dataset, shuffle=shuffle)
        dataloader = DataLoader(dataloader.dataset,
                                dataloader.batch_size,
                                sampler=sampler,
                                num_workers=dataloader.num_workers,
                                pin_memory=dataloader.pin_memory,
                                drop_last=dataloader.drop_last,
                                collate_fn=dataloader.collate_fn)

        dataloader.need_to_set_epoch = True
        return dataloader

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
        train_loader = self.on_reveive_dataloader(train_loader, mode='train')
        val_loader = self.on_reveive_dataloader(val_loader, mode='val')

        for t in range(num_epochs):
            if hasattr(train_loader, 'need_to_set_epoch'):
                train_loader.sampler.set_epoch(t)
            if hasattr(val_loader, 'need_to_set_epoch'):
                val_loader.sampler.set_epoch(t)

            num_batches = len(train_loader)

            # set model to train mode
            for k, m in self.modules.items():
                self.modules[k].train()

            # hook
            self.before_training_epoch(t + 1, num_epochs, **kwargs)
            for batch_id, train_batch in enumerate(train_loader):
                # hook
                self.before_training_step(batch_id, num_batches, **kwargs)
                # hook
                self.on_training_step(train_batch, batch_id, num_batches,
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
            with torch.no_grad():
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
        test_loader = self.on_reveive_dataloader(test_loader, mode='test')

        for k, m in self.modules.items():
            self.modules[k].eval()

        num_batches = len(test_loader)
        self.before_test_epoch(**kwargs)
        with torch.no_grad():
            for batch_id, test_batch in enumerate(test_loader):
                # hook
                self.before_test_step(batch_id, num_batches, **kwargs)
                # hook
                self.on_test_step(test_batch, batch_id, num_batches, **kwargs)
                # hook
                self.after_test_step(batch_id, num_batches, **kwargs)
        self.after_test_epoch(**kwargs)

    def before_test_epoch(self, **kwargs):
        # can be overwritten
        ...

    def before_test_step(self, batch_id: int, num_batches: int, **kwargs):
        # can be overwritten
        ...

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        raise NotImplementedError

    def after_test_step(self, batch_id: int, num_batches: int, **kwargs):
        # can be overwritten
        ...

    def after_test_epoch(self, **kwargs):
        # can be overwritten
        ...

    def load_state_dict(self, load_path, strict=False):
        if self.is_distributed:
            map_location = {'cpu': f'cuda:{self.local_rank}'}
        else:
            map_location = {'cpu': self.device}

        state_dict = torch.load(load_path, map_location=map_location)
        for k, m in self.modules.items():
            self.modules[k].load_state_dict(state_dict[k], strict=strict)

    def save_state_dict(self, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if self.is_main:
            state_dict = {}
            for k, m in self.modules.items():
                if isinstance(m, DistributedDataParallel):
                    m = m.module
                state_dict[k] = m.state_dict()
            torch.save(state_dict, save_path)

        # Use a barrier() to make sure that process a loads the model after process b
        # saves it (In case the user calls load_state_dict right after save_state_dict).
        if self.is_distributed:
            dist.barrier()