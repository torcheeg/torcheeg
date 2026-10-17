# Architecture adapted from Braindecode's Deep4Net.
# Copyright (c) 2017-2020 Braindecode Developers. BSD-3-Clause.
# See LICENSES/Braindecode.txt for the retained license.

import math
from numbers import Integral
from typing import Optional, Tuple
import warnings

import torch
from torch import nn

from .eegnet import Conv2dWithConstraint


def _pooled_length(length, kernel, pool_kernel, pool_stride, minimum=1):
    for _ in range(4):
        length = length - kernel + 1
        if length < minimum:
            return 0
        length = (length - pool_kernel) // pool_stride + 1
        if length < 1:
            return 0
    return length


def _convolution(*args, max_norm=None, **kwargs):
    if max_norm is None:
        return nn.Conv2d(*args, **kwargs)
    return Conv2dWithConstraint(*args, max_norm=max_norm, **kwargs)


class _DeepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pool_kernel,
                 pool_stride, dropout, batch_norm, momentum, max_norm):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.conv = _convolution(in_channels, out_channels, (1, kernel),
                                 bias=not batch_norm, max_norm=max_norm)
        self.norm = (nn.BatchNorm2d(out_channels, eps=1e-5, momentum=momentum)
                     if batch_norm else nn.Identity())
        self.activation = nn.ELU()
        self.pool = nn.MaxPool2d((1, pool_kernel), (1, pool_stride))

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        return self.pool(self.activation(self.norm(x)))


class DeepConvNet(nn.Module):
    r'''
    Four-block Deep ConvNet from Schirrmeister et al. (2017). A split temporal
    and spatial convolution is followed by three temporal blocks. Each block
    uses batch normalization, ELU and max pooling. A full-window convolution
    returns classification logits for :class:`torch.nn.CrossEntropyLoss`.

    - Paper: Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 2017, 38(11): 5391-5420.
    - URL: https://doi.org/10.1002/hbm.23730

    The default configuration follows Braindecode Deep4Net with split first
    layers, ELU, max pooling, a shared temporal kernel, automatic final
    convolution size, and no log-softmax. Dropout precedes blocks two to four.
    Construction does not run a dummy forward or update BatchNorm statistics.
    Models start in training mode, as other TorchEEG modules.

    For the EEG-Arena baseline, use ``temporal_bias=False``,
    ``conv_max_norm=2.0``, ``classifier_max_norm=0.5``,
    ``classifier_dropout=dropout``, ``initialization='pytorch'``, and
    ``auto_kernel=True``. Automatic adaptation reduces the shared kernel when
    needed to fit four blocks, keeping at least two time points after every
    convolution. A warning reports the actual kernel. The model raises an
    error if no kernel of at least two samples fits. Temporal kernels are
    measured in samples; the model does not infer a sampling rate.

    Trainable constrained weights are projected before each forward. Frozen
    weights use a functional projection and retain their stored values.

    .. code-block:: python

        import torch
        from torcheeg.models import DeepConvNet

        model = DeepConvNet(num_electrodes=22, chunk_size=1000,
                            num_classes=4)
        logits = model(torch.randn(2, 1, 22, 1000))
        # logits.shape == (2, 4)

    Args:
        num_electrodes (int): Number of input EEG channels. (default: :obj:`22`)
        chunk_size (int): Number of samples per input window. (default: :obj:`1000`)
        num_classes (int): Number of output classes. (default: :obj:`2`)
        temporal_filters (int): Number of first temporal filters. (default: :obj:`25`)
        spatial_filters (int): Number of first spatial filters. (default: :obj:`25`)
        hidden_channels (tuple, optional): Output filters of blocks two to four. None uses two, four and eight times spatial_filters. (default: :obj:`None`)
        temporal_kernel (int): Requested kernel length shared by all temporal convolutions. (default: :obj:`10`)
        pool_kernel (int): Max-pooling length in samples. (default: :obj:`3`)
        pool_stride (int): Max-pooling stride in samples. (default: :obj:`3`)
        batch_norm (bool): Whether to normalize convolution outputs. (default: :obj:`True`)
        batch_norm_momentum (float): Batch normalization momentum. (default: :obj:`0.1`)
        dropout (float): Dropout probability before blocks two to four. (default: :obj:`0.5`)
        classifier_dropout (float): Dropout probability before the classifier. (default: :obj:`0.0`)
        temporal_bias (bool): Whether the first temporal convolution has a bias. (default: :obj:`True`)
        conv_max_norm (float, optional): Maximum L2 norm per convolution output filter. None disables projection. (default: :obj:`None`)
        classifier_max_norm (float, optional): Maximum L2 norm per class weight vector. None disables projection. (default: :obj:`None`)
        initialization (str): Use :obj:`'xavier'` or native :obj:`'pytorch'` layer initialization. (default: :obj:`'xavier'`)
        auto_kernel (bool): Whether to reduce kernels explicitly for short input windows. (default: :obj:`False`)
    '''

    def __init__(self, num_electrodes: int = 22, chunk_size: int = 1000,
                 num_classes: int = 2, temporal_filters: int = 25,
                 spatial_filters: int = 25,
                 hidden_channels: Optional[Tuple[int, int, int]] = None,
                 temporal_kernel: int = 10, pool_kernel: int = 3,
                 pool_stride: int = 3, batch_norm: bool = True,
                 batch_norm_momentum: float = 0.1, dropout: float = 0.5,
                 classifier_dropout: float = 0.0, temporal_bias: bool = True,
                 conv_max_norm: Optional[float] = None,
                 classifier_max_norm: Optional[float] = None,
                 initialization: str = 'xavier', auto_kernel: bool = False):
        super().__init__()
        dimensions = dict(num_electrodes=num_electrodes, chunk_size=chunk_size,
            num_classes=num_classes, temporal_filters=temporal_filters,
            spatial_filters=spatial_filters, temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel, pool_stride=pool_stride)
        for name, value in dimensions.items():
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
            setattr(self, name, int(value))
        if hidden_channels is None:
            hidden_channels = tuple(self.spatial_filters * factor for factor in (2, 4, 8))
        if (not isinstance(hidden_channels, (tuple, list)) or len(hidden_channels) != 3
                or any(not isinstance(value, Integral)
                or isinstance(value, bool) or value < 1 for value in hidden_channels)):
            raise ValueError('hidden_channels must contain three positive integers.')
        self.hidden_channels = tuple(int(value) for value in hidden_channels)
        if not all(isinstance(value, bool) for value in (batch_norm, temporal_bias, auto_kernel)):
            raise ValueError('batch_norm, temporal_bias and auto_kernel must be booleans.')
        if not 0 <= batch_norm_momentum <= 1:
            raise ValueError('batch_norm_momentum must be between 0 and 1.')
        for name, value in [('conv_max_norm', conv_max_norm),
                            ('classifier_max_norm', classifier_max_norm)]:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive or None.')
        if initialization not in ('xavier', 'pytorch'):
            raise ValueError("initialization must be 'xavier' or 'pytorch'.")
        for name, value in [('dropout', dropout), ('classifier_dropout', classifier_dropout)]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and between 0 and 1.')
        self.requested_temporal_kernel = self.temporal_kernel
        self.auto_kernel = auto_kernel
        if auto_kernel:
            selected = None
            for kernel in range(min(self.temporal_kernel, self.chunk_size), 1, -1):
                if _pooled_length(self.chunk_size, kernel, self.pool_kernel,
                                  self.pool_stride, minimum=2):
                    selected = kernel
                    break
            if selected is None:
                raise ValueError('No temporal kernel of at least two samples fits four blocks.')
            self.temporal_kernel = selected
            if self.temporal_kernel != self.requested_temporal_kernel:
                warnings.warn(f'temporal_kernel reduced from {self.requested_temporal_kernel} '
                              f'to {self.temporal_kernel} for chunk_size={self.chunk_size}.',
                              UserWarning, stacklevel=2)
        self._pooled_size = _pooled_length(self.chunk_size, self.temporal_kernel,
                                         self.pool_kernel, self.pool_stride)
        if not self._pooled_size:
            raise ValueError('chunk_size does not fit four convolution/pooling blocks.')
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout
        self.temporal_bias = temporal_bias
        self.conv_max_norm = conv_max_norm
        self.classifier_max_norm = classifier_max_norm
        self.initialization = initialization
        self.temporal_conv = _convolution(1, self.temporal_filters,
            (1, self.temporal_kernel), bias=temporal_bias, max_norm=conv_max_norm)
        self.spatial_conv = _convolution(self.temporal_filters, self.spatial_filters,
            (self.num_electrodes, 1), bias=not batch_norm, max_norm=conv_max_norm)
        self.norm = (nn.BatchNorm2d(self.spatial_filters, eps=1e-5,
                                   momentum=batch_norm_momentum)
                     if batch_norm else nn.Identity())
        self.activation = nn.ELU()
        self.pool = nn.MaxPool2d((1, self.pool_kernel), (1, self.pool_stride))
        self.blocks = nn.ModuleList()
        in_channels = self.spatial_filters
        for out_channels in self.hidden_channels:
            self.blocks.append(_DeepConvBlock(in_channels, out_channels,
                self.temporal_kernel, self.pool_kernel, self.pool_stride,
                dropout, batch_norm, batch_norm_momentum, conv_max_norm))
            in_channels = out_channels
        self.drop = nn.Dropout(classifier_dropout)
        self.classifier = _convolution(in_channels, self.num_classes,
            (1, self._pooled_size), bias=True, max_norm=classifier_max_norm)
        if initialization == 'xavier':
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @property
    def feature_dim(self) -> int:
        return self.hidden_channels[-1] * self._pooled_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG windows shaped :obj:`[batch, 1, num_electrodes, chunk_size]`.

        Returns:
            torch.Tensor[batch, num_classes]: Classification logits.
        '''
        if (x.ndim != 4 or x.shape[1] != 1
                or x.shape[2] != self.num_electrodes or x.shape[3] != self.chunk_size):
            raise ValueError('Expected [batch, 1, num_electrodes, chunk_size] with the configured dimensions.')
        x = self.pool(self.activation(self.norm(self.spatial_conv(self.temporal_conv(x)))))
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.drop(x)).flatten(start_dim=1)
