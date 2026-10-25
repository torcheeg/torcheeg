# Adapted from Braindecode v1.3.0 (BSD-3-Clause).
# Authors: Bruno Aristimunha and Cedric Rommel.
# See LICENSES/EEGInceptionERP.txt for the complete notice.

import math
from numbers import Integral, Real
from typing import Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F


class _TemporalConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit asymmetric SAME padding also supports older PyTorch releases.
        padding = self.kernel_size[1] - 1
        return super().forward(F.pad(x, (padding // 2, padding - padding // 2)))


class _InceptionBlock(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class EEGInceptionERP(nn.Module):
    r"""EEG-Inception for event-related potential classification.

    - Paper: Santamaria-Vazquez E, Martinez-Cagigal V, Vaquerizo-Villar F,
      et al. EEG-Inception: A Novel Deep Convolutional Neural Network for
      Assistive ERP-Based Brain-Computer Interfaces. IEEE TNSRE, 2020.
    - URL: https://doi.org/10.1109/TNSRE.2020.3048106
    - Related Project: https://github.com/braindecode/braindecode

    Retains both three-branch Inception blocks, branch-specific depthwise
    spatial filters, the two-convolution aggregation block and dense readout.
    Outputs are logits for cross entropy. This supervised model has no
    pretrained foundation checkpoint. It does not filter or resample data.

    .. code-block:: python

        from torcheeg.models import EEGInceptionERP
        import torch

        model = EEGInceptionERP(num_electrodes=8, chunk_size=128,
                                sampling_rate=128, num_classes=2)
        logits = model(torch.randn(4, 8, 128))

    Args:
        num_electrodes (int): Number of electrodes. (default: :obj:`8`)
        chunk_size (int): Number of time samples per epoch. The default is a
            one-second epoch at 128 Hz. (default: :obj:`128`)
        num_classes (int): Number of output classes. (default: :obj:`2`)
        sampling_rate (float): Sampling rate used to convert temporal scales
            to kernel lengths. Must match the data. (default: :obj:`128`)
        num_filters (int): Filters per temporal branch, at least two.
            (default: :obj:`8`)
        scales (tuple): Exactly three temporal scales in seconds. Kernel
            lengths use integer truncation. Each must yield at least four
            samples. (default: :obj:`(0.5, 0.25, 0.125)`)
        depth_multiplier (int): Spatial filters per temporal feature.
            (default: :obj:`2`)
        pool_sizes (tuple): Four temporal pooling factors. The second
            Inception block retains the reference's kernel-length division
            by four, even with custom pooling. (default: :obj:`(4, 2, 2, 2)`)
        dropout (float): Dropout probability. (default: :obj:`0.5`)
        batch_norm_momentum (float): Batch normalization momentum.
            (default: :obj:`0.01`)
        activation (type): An :class:`~torch.nn.Module` class constructed
            without arguments. (default: :class:`~torch.nn.ELU`)
    """

    def __init__(
        self,
        num_electrodes: int = 8,
        chunk_size: int = 128,
        num_classes: int = 2,
        sampling_rate: float = 128,
        num_filters: int = 8,
        scales: Tuple[float, float, float] = (0.5, 0.25, 0.125),
        depth_multiplier: int = 2,
        pool_sizes: Tuple[int, int, int, int] = (4, 2, 2, 2),
        dropout: float = 0.5,
        batch_norm_momentum: float = 0.01,
        activation: Type[nn.Module] = nn.ELU,
    ):
        super().__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('num_filters', num_filters),
            ('depth_multiplier', depth_multiplier),
        ]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if num_filters < 2:
            raise ValueError('num_filters must be at least two.')
        if (
            not isinstance(sampling_rate, Real)
            or isinstance(sampling_rate, bool)
            or not math.isfinite(sampling_rate)
            or sampling_rate <= 0
        ):
            raise ValueError('sampling_rate must be positive and finite.')
        if (
            not isinstance(scales, (tuple, list))
            or len(scales) != 3
            or any(
                not isinstance(s, Real)
                or isinstance(s, bool)
                or not math.isfinite(s)
                or s <= 0
                for s in scales
            )
        ):
            raise ValueError('scales must contain three positive finite durations.')
        lengths = tuple(s * sampling_rate for s in scales)
        if any(not math.isfinite(s) or s < 4 for s in lengths):
            raise ValueError('Each temporal scale must yield at least four samples.')
        if (
            not isinstance(pool_sizes, (tuple, list))
            or len(pool_sizes) != 4
            or any(
                not isinstance(p, Integral) or isinstance(p, bool) or p < 1
                for p in pool_sizes
            )
        ):
            raise ValueError('pool_sizes must contain four positive integers.')
        if chunk_size < math.prod(pool_sizes):
            raise ValueError('chunk_size is too short for all four pooling layers.')
        for name, value in [
            ('dropout', dropout),
            ('batch_norm_momentum', batch_norm_momentum),
        ]:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f'{name} must be between zero and one.')
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise ValueError('activation must be a torch.nn.Module class.')
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.num_filters = num_filters
        self.scales = tuple(scales)
        self.depth_multiplier = depth_multiplier
        self.pool_sizes = tuple(pool_sizes)
        self.dropout = dropout
        self.batch_norm_momentum = batch_norm_momentum
        kernels = tuple(int(s) for s in lengths)

        def temporal_block(in_channels, out_channels, kernel, bias=False):
            return nn.Sequential(
                _TemporalConv2d(in_channels, out_channels, (1, kernel), bias=bias),
                nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum),
                activation(),
                nn.Dropout(dropout),
            )

        branches = []
        for kernel in kernels:
            branch = temporal_block(1, num_filters, kernel, bias=True)
            branch.add_module(
                '4',
                nn.Conv2d(
                    num_filters,
                    num_filters * depth_multiplier,
                    (num_electrodes, 1),
                    groups=num_filters,
                    bias=False,
                ),
            )
            branch.add_module(
                '5',
                nn.BatchNorm2d(
                    num_filters * depth_multiplier, momentum=batch_norm_momentum
                ),
            )
            branch.add_module('6', activation())
            branch.add_module('7', nn.Dropout(dropout))
            branches.append(branch)
        self.inception_block_1 = _InceptionBlock(branches)
        self.avg_pool_1 = nn.AvgPool2d((1, pool_sizes[0]))
        concatenated = 3 * num_filters
        self.inception_block_2 = _InceptionBlock(
            [
                temporal_block(
                    concatenated * depth_multiplier, num_filters, kernel // 4
                )
                for kernel in kernels
            ]
        )
        self.avg_pool_2 = nn.AvgPool2d((1, pool_sizes[1]))
        self.final_block = nn.Sequential(
            _TemporalConv2d(concatenated, concatenated // 2, (1, 8), bias=False),
            nn.BatchNorm2d(concatenated // 2, momentum=batch_norm_momentum),
            activation(),
            nn.Dropout(dropout),
            nn.AvgPool2d((1, pool_sizes[2])),
            _TemporalConv2d(concatenated // 2, concatenated // 4, (1, 4), bias=False),
            nn.BatchNorm2d(concatenated // 4, momentum=batch_norm_momentum),
            activation(),
            nn.Dropout(dropout),
            nn.AvgPool2d((1, pool_sizes[3])),
        )
        self.flat = nn.Flatten()
        self.final_layer = nn.Sequential()
        self.final_layer.add_module(
            'fc',
            nn.Linear(
                (chunk_size // math.prod(pool_sizes)) * (concatenated // 4), num_classes
            ),
        )
        self.final_layer.add_module('identity', nn.Identity())
        # Match v1.3.0's actual initializer: retain default weights, zero biases.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Classify epochs with shape :obj:`[batch, electrodes, time]` or
        :obj:`[batch, 1, electrodes, time]`.

        Returns:
            torch.Tensor: Logits with shape :obj:`[batch, num_classes]`.
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
        x = self.avg_pool_1(self.inception_block_1(x))
        x = self.avg_pool_2(self.inception_block_2(x))
        return self.final_layer(self.flat(self.final_block(x)))
