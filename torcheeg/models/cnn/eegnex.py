# Adapted from Braindecode v1.3.0 (BSD-3-Clause).
# Author: Bruno Aristimunha. See LICENSES/EEGNeX.txt.

import math
from numbers import Integral, Real
from typing import Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F


class _TemporalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_same = tuple(int(v) for v in torch.__version__.split('.')[:2]) >= (
            1,
            10,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._native_same:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                'same',
                self.dilation,
                self.groups,
            )
        padding = (self.kernel_size[1] - 1) * self.dilation[1]
        return super().forward(F.pad(x, (padding // 2, padding - padding // 2)))


class _MaxNorm(nn.Module):
    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.renorm(p=2, dim=0, maxnorm=self.max_norm)


class EEGNeX(nn.Module):
    r"""EEGNeX for EEG classification with differentiable weight constraints.

    - Paper: Chen X, Teng X, Chen H, et al. Toward reliable signals decoding
      for electroencephalogram: A benchmark study to EEGNeX. Biomedical
      Signal Processing and Control, 87:105475, 2024.
    - Related Project: https://github.com/chenxiachan/EEGNeX
    - Reference: https://github.com/braindecode/braindecode

    Retains five convolutional blocks, depthwise spatial filtering, dilated
    temporal convolutions and max-norm parametrizations on the spatial and
    dense weights. The unconstrained original parameters remain trainable;
    constraints are applied differentiably when weights are accessed.
    Returns logits for cross entropy. No pretrained weights or preprocessing
    are included. Requires :mod:`torch.nn.utils.parametrize` (PyTorch 1.9+).

    .. code-block:: python

        from torcheeg.models import EEGNeX
        import torch

        model = EEGNeX(num_electrodes=22, chunk_size=1000, num_classes=4)
        logits = model(torch.randn(3, 22, 1000))

    Args:
        num_electrodes (int): Number of electrodes. (default: :obj:`22`)
        chunk_size (int): Number of samples per epoch. (default: :obj:`1000`)
        num_classes (int): Number of output classes. (default: :obj:`4`)
        num_filters (tuple): Filters in the first and second blocks.
            (default: :obj:`(8, 32)`)
        depth_multiplier (int): Spatial filters per second-block feature.
            (default: :obj:`2`)
        temporal_kernels (tuple): Kernel lengths for the shared first/second
            blocks, fourth block and fifth block, in samples.
            (default: :obj:`(64, 16, 16)`)
        dilations (tuple): Temporal dilation in blocks four and five.
            (default: :obj:`(2, 4)`)
        pool_sizes (tuple): Temporal average-pool factors. Each pooling layer
            uses one padded sample on each side. (default: :obj:`(4, 8)`)
        dropout (float): Dropout probability. (default: :obj:`0.5`)
        max_norm_conv (float): Maximum spatial-filter L2 norm.
            (default: :obj:`1.0`)
        max_norm_linear (float): Maximum readout-row L2 norm.
            (default: :obj:`0.25`)
        activation (type): Module class constructed without arguments.
            (default: :class:`~torch.nn.ELU`)
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 1000,
        num_classes: int = 4,
        num_filters: Tuple[int, int] = (8, 32),
        depth_multiplier: int = 2,
        temporal_kernels: Tuple[int, int, int] = (64, 16, 16),
        dilations: Tuple[int, int] = (2, 4),
        pool_sizes: Tuple[int, int] = (4, 8),
        dropout: float = 0.5,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
        activation: Type[nn.Module] = nn.ELU,
    ):
        super().__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('depth_multiplier', depth_multiplier),
        ]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name, values, count in [
            ('num_filters', num_filters, 2),
            ('temporal_kernels', temporal_kernels, 3),
            ('dilations', dilations, 2),
            ('pool_sizes', pool_sizes, 2),
        ]:
            if (
                not isinstance(values, (tuple, list))
                or len(values) != count
                or any(
                    not isinstance(v, Integral) or isinstance(v, bool) or v < 1
                    for v in values
                )
            ):
                raise ValueError(f'{name} must contain {count} positive integers.')
        if any(p < 2 for p in pool_sizes):
            raise ValueError('pool_sizes must be at least two for one-sample padding.')
        length = (chunk_size + 2) // pool_sizes[0]
        length = (length + 2) // pool_sizes[1] if length > 0 else 0
        if length < 1:
            raise ValueError('chunk_size is too short for both padded pooling layers.')
        if (
            not isinstance(dropout, Real)
            or isinstance(dropout, bool)
            or not math.isfinite(dropout)
            or not 0 <= dropout <= 1
        ):
            raise ValueError('dropout must be between zero and one.')
        for name, value in [
            ('max_norm_conv', max_norm_conv),
            ('max_norm_linear', max_norm_linear),
        ]:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f'{name} must be positive and finite.')
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise ValueError('activation must be a torch.nn.Module class.')
        # Keep older TorchEEG model imports usable without this newer API.
        try:
            from torch.nn.utils.parametrize import register_parametrization
        except ImportError as error:
            raise ImportError(
                'EEGNeX requires torch.nn.utils.parametrize (PyTorch 1.9+).'
            ) from error

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_filters = tuple(num_filters)
        self.depth_multiplier = depth_multiplier
        self.temporal_kernels = tuple(temporal_kernels)
        self.dilations = tuple(dilations)
        self.pool_sizes = tuple(pool_sizes)
        self.dropout = dropout
        self.max_norm_conv = max_norm_conv
        self.max_norm_linear = max_norm_linear
        first, second = num_filters
        spatial = second * depth_multiplier
        self.block_1 = nn.Sequential(
            nn.Identity(),
            _TemporalConv2d(1, first, (1, temporal_kernels[0]), bias=False),
            nn.BatchNorm2d(first),
        )
        self.block_2 = nn.Sequential(
            _TemporalConv2d(first, second, (1, temporal_kernels[0]), bias=False),
            nn.BatchNorm2d(second),
        )
        spatial_conv = nn.Conv2d(
            second, spatial, (num_electrodes, 1), groups=second, bias=False
        )
        # The reference initializes only this constrained convolution with Xavier.
        nn.init.xavier_uniform_(spatial_conv.weight, gain=1)
        register_parametrization(spatial_conv, 'weight', _MaxNorm(max_norm_conv))
        self.block_3 = nn.Sequential(
            spatial_conv,
            nn.BatchNorm2d(spatial),
            activation(),
            nn.AvgPool2d((1, pool_sizes[0]), padding=(0, 1)),
            nn.Dropout(dropout),
        )
        self.block_4 = nn.Sequential(
            _TemporalConv2d(
                spatial,
                second,
                (1, temporal_kernels[1]),
                dilation=(1, dilations[0]),
                bias=False,
            ),
            nn.BatchNorm2d(second),
        )
        self.block_5 = nn.Sequential(
            _TemporalConv2d(
                second,
                first,
                (1, temporal_kernels[2]),
                dilation=(1, dilations[1]),
                bias=False,
            ),
            nn.BatchNorm2d(first),
            activation(),
            nn.AvgPool2d((1, pool_sizes[1]), padding=(0, 1)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.final_layer = nn.Linear(first * length, num_classes)
        register_parametrization(self.final_layer, 'weight', _MaxNorm(max_norm_linear))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return class logits for :obj:`[batch, electrodes, time]` or
        :obj:`[batch, 1, electrodes, time]` epochs.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if (
            x.ndim != 4
            or x.shape[1] != 1
            or x.shape[2] != self.num_electrodes
            or x.shape[3] != self.chunk_size
        ):
            raise ValueError(
                'Expected [batch, electrodes, time] or '
                '[batch, 1, electrodes, time] matching the constructor.'
            )
        x = self.block_3(self.block_2(self.block_1(x)))
        return self.final_layer(self.block_5(self.block_4(x)))
