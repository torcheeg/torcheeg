# Adapted from Braindecode's BSD-3-Clause MSVTNet implementation.
# Authors: Tao Yang and Bruno Aristimunha. See LICENSES/MSVTNet.txt.

import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def _probability(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f'{name} must be finite and between zero and one.')
    return float(value)


class _SamePadConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        version = tuple(int(v) for v in torch.__version__.split('.')[:2])
        self._native_same = version >= (1, 10)

    def forward(self, x):
        # Match the author's numerical path on supported PyTorch versions.
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
        total = self.kernel_size[-1] - 1
        return super().forward(F.pad(x, (total // 2, total - total // 2, 0, 0)))


class _TSConv(nn.Sequential):
    def __init__(
        self,
        num_electrodes,
        num_filters,
        temporal_kernel,
        depthwise_kernel,
        depth_multiplier,
        pool_size1,
        pool_size2,
        dropout,
    ):
        width = num_filters * depth_multiplier
        super().__init__(
            _SamePadConv2d(1, num_filters, (1, temporal_kernel), bias=False),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(
                num_filters, width, (num_electrodes, 1), groups=num_filters, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size1)),
            nn.Dropout(dropout),
            _SamePadConv2d(
                width, width, (1, depthwise_kernel), groups=width, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size2)),
            nn.Dropout(dropout),
        )


class _PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, hid_channels):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, sequence_length, hid_channels))

    def forward(self, x):
        return x + self.pe


class _Transformer(nn.Module):
    def __init__(
        self, sequence_length, hid_channels, heads, mlp_channels, dropout, depth
    ):
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, hid_channels))
        self.pos_embedding = _PositionalEncoding(sequence_length + 1, hid_channels)
        self.dropout = nn.Dropout(dropout)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hid_channels,
                heads,
                mlp_channels,
                dropout,
                batch_first=True,
                norm_first=True,
            ),
            depth,
            norm=nn.LayerNorm(hid_channels),
        )

    def forward(self, x):
        x = torch.cat((self.cls_embedding.expand(x.shape[0], -1, -1), x), dim=1)
        return self.trans(self.dropout(self.pos_embedding(x)))[:, 0]


class MSVTNet(nn.Module):
    r"""Multi-scale vision transformer for EEG motor imagery decoding.

    Liu et al. MSVTNet: Multi-Scale Vision Transformer Neural Network for
    EEG-Based Motor Imagery Decoding. IEEE JBHI, 2024.
    `Paper <https://doi.org/10.1109/JBHI.2024.3450753>`_.

    The complete convolution branches, auxiliary classifiers, class token and
    transformer are retained. Outputs are logits; use :class:`MSVTNetLoss`
    with ``return_branch_logits=True`` for the author's joint training loss.
    With the default output, standard classifier trainers use the main head.
    This supervised architecture does not load foundation-model weights.

    Args:
        num_electrodes (int): Number of input electrodes. (default: :obj:`22`)
        chunk_size (int): Number of time samples. (default: :obj:`1000`)
        num_classes (int): Number of classes. (default: :obj:`4`)
        num_filters (sequence of int): Filters in each branch.
            (default: :obj:`(9, 9, 9, 9)`)
        temporal_kernels (sequence of int): Temporal kernels in each branch.
            (default: :obj:`(15, 31, 63, 125)`)
        depthwise_kernel (int): Second temporal kernel. (default: :obj:`15`)
        depth_multiplier (int): Spatial depth multiplier. (default: :obj:`2`)
        pool_size1 (int): First average pooling width. (default: :obj:`8`)
        pool_size2 (int): Second average pooling width. (default: :obj:`7`)
        dropout (float): Convolutional dropout probability. (default: :obj:`0.3`)
        heads (int): Attention heads. (default: :obj:`8`)
        mlp_ratio (float): Transformer feedforward width ratio. (default: :obj:`1`)
        attn_dropout (float): Transformer dropout probability. (default: :obj:`0.5`)
        depth (int): Transformer layers. (default: :obj:`2`)
        return_branch_logits (bool): Return main and auxiliary logits.
            (default: :obj:`False`)

    .. code-block:: python

        model = MSVTNet(num_electrodes=22, chunk_size=1000, num_classes=4)
        logits = model(torch.randn(2, 22, 1000))
        # logits.shape: (2, 4)

    Native parameter names are preserved, including ``last_head``. The
    author's parameter-free LogSoftmax is replaced by the logits contract.
    Constructor dimensions are computed analytically, without random probe
    forwards that update BatchNorm statistics.
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 1000,
        num_classes: int = 4,
        num_filters: Sequence[int] = (9, 9, 9, 9),
        temporal_kernels: Sequence[int] = (15, 31, 63, 125),
        depthwise_kernel: int = 15,
        depth_multiplier: int = 2,
        pool_size1: int = 8,
        pool_size2: int = 7,
        dropout: float = 0.3,
        heads: int = 8,
        mlp_ratio: float = 1,
        attn_dropout: float = 0.5,
        depth: int = 2,
        return_branch_logits: bool = False,
    ):
        super().__init__()
        self.num_electrodes = _positive_int('num_electrodes', num_electrodes)
        self.chunk_size = _positive_int('chunk_size', chunk_size)
        num_classes = _positive_int('num_classes', num_classes)
        depthwise_kernel = _positive_int('depthwise_kernel', depthwise_kernel)
        depth_multiplier = _positive_int('depth_multiplier', depth_multiplier)
        pool_size1 = _positive_int('pool_size1', pool_size1)
        pool_size2 = _positive_int('pool_size2', pool_size2)
        heads = _positive_int('heads', heads)
        depth = _positive_int('depth', depth)
        dropout = _probability('dropout', dropout)
        attn_dropout = _probability('attn_dropout', attn_dropout)
        num_filters = tuple(_positive_int('num_filters', v) for v in num_filters)
        temporal_kernels = tuple(
            _positive_int('temporal_kernels', v) for v in temporal_kernels
        )
        if not num_filters or len(num_filters) != len(temporal_kernels):
            raise ValueError(
                'num_filters and temporal_kernels need equal nonzero lengths.'
            )
        if not isinstance(return_branch_logits, bool):
            raise ValueError('return_branch_logits must be a bool.')
        self.return_branch_logits = return_branch_logits
        sequence_length = chunk_size // pool_size1 // pool_size2
        if sequence_length < 1:
            raise ValueError('chunk_size must be at least pool_size1 * pool_size2.')
        hid_channels = sum(num_filters) * depth_multiplier
        if hid_channels % heads:
            raise ValueError('The combined branch width must be divisible by heads.')
        if (
            isinstance(mlp_ratio, bool)
            or not isinstance(mlp_ratio, Real)
            or not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or not math.isfinite(hid_channels * mlp_ratio)
            or int(hid_channels * mlp_ratio) < 1
        ):
            raise ValueError(
                'mlp_ratio must produce a positive finite feedforward width.'
            )
        self.mstsconv = nn.ModuleList(
            [
                nn.Sequential(
                    _TSConv(
                        num_electrodes,
                        width,
                        kernel,
                        depthwise_kernel,
                        depth_multiplier,
                        pool_size1,
                        pool_size2,
                        dropout,
                    ),
                    Rearrange('batch channels 1 time -> batch time channels'),
                )
                for width, kernel in zip(num_filters, temporal_kernels)
            ]
        )
        self.branch_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(sequence_length * width * depth_multiplier, num_classes),
                )
                for width in num_filters
            ]
        )
        self.transformer = _Transformer(
            sequence_length,
            hid_channels,
            heads,
            int(hid_channels * mlp_ratio),
            attn_dropout,
            depth,
        )
        self.last_head = nn.Sequential(
            nn.Flatten(), nn.Linear(hid_channels, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """Map ``[batch, electrodes, time]`` or ``[batch, 1, electrodes, time]``
        to logits, optionally paired with a tuple of auxiliary logits.
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
                'Input must match the configured electrodes and chunk_size.'
            )
        branches = [block(x) for block in self.mstsconv]
        logits = self.last_head(self.transformer(torch.cat(branches, dim=2)))
        if self.return_branch_logits:
            return logits, tuple(
                head(value) for head, value in zip(self.branch_head, branches)
            )
        return logits


class MSVTNetLoss(nn.Module):
    r"""Joint main/branch cross entropy for :class:`MSVTNet`.

    The loss is ``main_weight * CE(main) + (1 - main_weight) * sum(CE(branch))``.
    Auxiliary losses are summed, not averaged. Pass the model's logits tuple
    and integer class targets, as returned with ``return_branch_logits=True``.

    Args:
        main_weight (float): Weight of the main classifier. (default: :obj:`0.6`)
    """

    def __init__(self, main_weight: float = 0.6):
        super().__init__()
        self.main_weight = _probability('main_weight', main_weight)

    def forward(self, outputs, targets):
        main, branches = outputs
        if not isinstance(branches, (list, tuple)) or not branches:
            raise ValueError('Provide a nonempty sequence of branch logits.')
        return (
            self.main_weight * F.cross_entropy(main, targets)
            + (1 - self.main_weight)
            * torch.stack(
                [F.cross_entropy(branch, targets) for branch in branches]
            ).sum()
        )
