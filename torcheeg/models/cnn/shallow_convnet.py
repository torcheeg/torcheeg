# Architecture adapted from Braindecode's ShallowFBCSPNet.
# Copyright (c) 2017-2020 Braindecode Developers. BSD-3-Clause.
# See LICENSES/Braindecode.txt for the retained license.

import math
from numbers import Integral
from typing import Optional

import torch
from torch import nn

from .eegnet import Conv2dWithConstraint


class ShallowConvNet(nn.Module):
    r'''
    Shallow ConvNet from Schirrmeister et al. (2017). Temporal and spatial
    convolutions are followed by batch normalization, squaring, mean pooling,
    a safe logarithm, dropout, and a full-window convolutional classifier.

    - Paper: Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 2017, 38(11): 5391-5420.
    - URL: https://doi.org/10.1002/hbm.23730

    This implements the split temporal/spatial, mean-pooling architecture of
    Braindecode's ShallowFBCSPNet with an automatically sized final convolution.
    It returns logits for :class:`torch.nn.CrossEntropyLoss`. It does not apply
    log-softmax, use cropped prediction, or infer a sampling rate. Set the
    input window and channel order to match the preprocessing protocol.

    For the EEG-Arena constrained baseline, set ``temporal_bias=False``,
    ``conv_max_norm=2.0``, ``classifier_max_norm=0.5``, and
    ``initialization='pytorch'``. These settings
    retain the square/mean/log feature extractor and constrain each output
    filter or class weight vector. Trainable weights are projected before
    each forward; frozen weights use a functional projection without changing
    stored parameters. Input layout remains the TorchEEG layout below.

    .. code-block:: python

        import torch
        from torcheeg.models import ShallowConvNet

        model = ShallowConvNet(num_electrodes=22, chunk_size=1000,
                               num_classes=4)
        logits = model(torch.randn(2, 1, 22, 1000))
        # logits.shape == (2, 4)

    Args:
        num_electrodes (int): Number of input EEG channels. (default: :obj:`22`)
        chunk_size (int): Number of samples in each input window. (default: :obj:`1000`)
        num_classes (int): Number of output classes. (default: :obj:`2`)
        temporal_filters (int): Number of temporal convolution filters. (default: :obj:`40`)
        spatial_filters (int): Number of spatial convolution filters. (default: :obj:`40`)
        temporal_kernel (int): Temporal kernel length in samples. (default: :obj:`25`)
        pool_kernel (int): Mean-pooling kernel length in samples. (default: :obj:`75`)
        pool_stride (int): Mean-pooling stride in samples. (default: :obj:`15`)
        batch_norm (bool): Whether to normalize spatial convolution outputs. (default: :obj:`True`)
        batch_norm_momentum (float): Batch normalization momentum. (default: :obj:`0.1`)
        dropout (float): Dropout probability. (default: :obj:`0.5`)
        log_eps (float): Positive lower bound before the logarithm. (default: :obj:`1e-6`)
        temporal_bias (bool): Whether the temporal convolution has a bias. (default: :obj:`True`)
        conv_max_norm (float, optional): Maximum L2 norm per temporal/spatial output filter. None disables projection. (default: :obj:`None`)
        classifier_max_norm (float, optional): Maximum L2 norm per class weight vector. None disables projection. (default: :obj:`None`)
        initialization (str): Use Xavier initialization or native PyTorch layer initialization. Options are :obj:`'xavier'` and :obj:`'pytorch'`. (default: :obj:`'xavier'`)
    '''

    def __init__(self, num_electrodes: int = 22, chunk_size: int = 1000,
                 num_classes: int = 2, temporal_filters: int = 40,
                 spatial_filters: int = 40, temporal_kernel: int = 25,
                 pool_kernel: int = 75, pool_stride: int = 15,
                 batch_norm: bool = True, batch_norm_momentum: float = 0.1,
                 dropout: float = 0.5, log_eps: float = 1e-6,
                 temporal_bias: bool = True, conv_max_norm: Optional[float] = None,
                 classifier_max_norm: Optional[float] = None,
                 initialization: str = 'xavier'):
        super().__init__()
        dimensions = dict(num_electrodes=num_electrodes, chunk_size=chunk_size,
                          num_classes=num_classes, temporal_filters=temporal_filters,
                          spatial_filters=spatial_filters, temporal_kernel=temporal_kernel,
                          pool_kernel=pool_kernel, pool_stride=pool_stride)
        for name, value in dimensions.items():
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
            setattr(self, name, int(value))
        if not isinstance(batch_norm, bool) or not isinstance(temporal_bias, bool):
            raise ValueError('batch_norm and temporal_bias must be booleans.')
        for name, value in [('conv_max_norm', conv_max_norm),
                            ('classifier_max_norm', classifier_max_norm)]:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive or None.')
        if not 0 <= batch_norm_momentum <= 1:
            raise ValueError('batch_norm_momentum must be between 0 and 1.')
        if not math.isfinite(log_eps) or log_eps <= 0:
            raise ValueError('log_eps must be finite and positive.')
        if initialization not in ('xavier', 'pytorch'):
            raise ValueError("initialization must be 'xavier' or 'pytorch'.")
        self.initialization = initialization
        self.log_eps = float(log_eps)
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout = dropout
        self.temporal_bias = temporal_bias
        self.conv_max_norm = conv_max_norm
        self.classifier_max_norm = classifier_max_norm
        self._pooled_size = (self.chunk_size - self.temporal_kernel + 1 - self.pool_kernel) // self.pool_stride + 1
        if self.chunk_size < self.temporal_kernel + self.pool_kernel - 1:
            raise ValueError('chunk_size must fit the temporal convolution and at least one pooling window.')

        convolution = nn.Conv2d if conv_max_norm is None else Conv2dWithConstraint
        constraint = {} if conv_max_norm is None else {'max_norm': conv_max_norm}
        self.temporal_conv = convolution(1, self.temporal_filters,
            (1, self.temporal_kernel), bias=temporal_bias, **constraint)
        self.spatial_conv = convolution(self.temporal_filters, self.spatial_filters,
            (self.num_electrodes, 1), bias=not batch_norm, **constraint)
        self.norm = (nn.BatchNorm2d(self.spatial_filters, eps=1e-5,
                                   momentum=batch_norm_momentum)
                     if batch_norm else nn.Identity())
        self.pool = nn.AvgPool2d((1, self.pool_kernel), (1, self.pool_stride))
        self.drop = nn.Dropout(dropout)
        classifier = nn.Conv2d if classifier_max_norm is None else Conv2dWithConstraint
        constraint = {} if classifier_max_norm is None else {'max_norm': classifier_max_norm}
        self.classifier = classifier(self.spatial_filters, self.num_classes,
                                     (1, self._pooled_size), bias=True, **constraint)
        if initialization == 'xavier':
            for module in (self.temporal_conv, self.spatial_conv, self.classifier):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def feature_dim(self) -> int:
        return self.spatial_filters * self._pooled_size

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
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.norm(x).square()
        x = self.pool(x).clamp_min(self.log_eps).log()
        x = self.drop(x)
        return self.classifier(x).flatten(start_dim=1)
