# Adapted from Braindecode's Apache-2.0 MSCFormer implementation.
# Authors: Wei Zhao and LiQing. Attention/residual helpers also retain
# Braindecode's BSD-3-Clause notice. See LICENSES/MSCFormer*.txt.

import math
from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability


class _MSCAttention(nn.Module):
    def __init__(self, width, heads, dropout, scale):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.scale = scale
        self.keys = nn.Linear(width, width)
        self.queries = nn.Linear(width, width)
        self.values = nn.Linear(width, width)
        self.att_drop = dropout
        self.projection = nn.Linear(width, width)
        self._native_attention = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        batch, length, width = x.shape

        def split(layer):
            return (
                layer(x)
                .reshape(batch, length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        queries, keys, values = (
            split(self.queries),
            split(self.keys),
            split(self.values),
        )
        queries = queries * (self.scale * self.head_dim**0.5)
        dropout = self.att_drop if self.training else 0.0
        # Preserve the reference CPU dropout math and random draw order.
        if (
            dropout > 0
            and x.device.type == 'cpu'
            and x.dtype in (torch.float32, torch.float64)
        ) or not self._native_attention:
            half = math.sqrt(1.0 / math.sqrt(self.head_dim))
            weights = torch.matmul(
                queries * half, keys.transpose(-2, -1) * half
            ).softmax(-1)
            weights = F.dropout(weights, dropout, self.training)
            out = torch.matmul(weights, values)
        else:
            out = F.scaled_dot_product_attention(
                queries, keys, values, dropout_p=dropout
            )
        return self.projection(out.transpose(1, 2).reshape(batch, length, width))


class _MSCResidual(nn.Module):
    def __init__(self, module, width, dropout):
        super().__init__()
        self.module = module
        self.drop = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(width)

    def forward(self, x):
        return self.layernorm(self.drop(self.module(x)) + x)


class _MSCTransformerBlock(nn.Module):
    def __init__(self, width, heads, expansion, dropout, scale):
        super().__init__()
        attention = _MSCAttention(width, heads, dropout, scale)
        feed_forward = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
        )
        self.attention = _MSCResidual(attention, width, dropout)
        self.feed_forward = _MSCResidual(feed_forward, width, dropout)

    def forward(self, x):
        return self.feed_forward(self.attention(x))


class _MSCPosition(nn.Module):
    def __init__(self, length, width, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, width))

    def forward(self, x):
        return self.dropout(x + self.encoding[:, : x.size(1), :])


class _MSCPatchEmbedding(nn.Module):
    def __init__(self, electrodes, filters, kernels, pool_size, dropout):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, filters, (1, kernel)),
                    nn.Conv2d(filters, filters, (electrodes, 1), groups=filters),
                    nn.BatchNorm2d(filters),
                    nn.ELU(),
                    nn.AvgPool2d((1, pool_size)),
                    nn.Dropout(dropout),
                )
                for kernel in kernels
            ]
        )
        version = tuple(int(v) for v in torch.__version__.split('.')[:2])
        self._native_same = version >= (1, 10)

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            temporal, spatial = branch[0], branch[1]
            # Both original linear convolutions retain every weight and bias.
            weight = spatial.weight.squeeze(1) * temporal.weight.squeeze(1)
            bias = torch.addcmul(
                spatial.bias, spatial.weight.sum((1, 2, 3)), temporal.bias
            )
            if self._native_same:
                out = F.conv1d(x, weight, bias, padding='same').unsqueeze(2)
            else:
                total = weight.shape[-1] - 1
                padded = F.pad(x, (total // 2, total - total // 2))
                out = F.conv1d(padded, weight, bias).unsqueeze(2)
            for index, layer in enumerate(branch):
                if index > 1:
                    out = layer(out)
            outputs.append(out)
        return torch.cat(outputs, dim=1).squeeze(2).transpose(1, 2)


class MSCFormer(nn.Module):
    r"""Complete multi-scale convolutional Transformer for motor imagery.

    Zhao et al. Multi-scale convolutional transformer network for motor
    imagery brain-computer interface. Scientific Reports, 2025.
    `Paper <https://doi.org/10.1038/s41598-025-96611-5>`_.

    Preserves all three temporal/spatial convolution branches, five post-norm
    Transformer blocks, learned positions, zero class token and full readout
    of the pinned Braindecode adaptation. Raw floating inputs are
    :obj:`[batch, electrodes, time]` or :obj:`[batch, 1, electrodes, time]`.
    Sampling rate, units, channel order and preprocessing belong to the caller.
    Outputs are logits for cross entropy; no pretrained weights are loaded.

    Args:
        num_electrodes (int): Ordered input rows. (default: :obj:`22`)
        chunk_size (int): Input time samples. (default: :obj:`1000`)
        num_classes (int): Output classes. (default: :obj:`4`)
        num_filters (int): Filters per convolution branch. (default: :obj:`16`)
        temporal_kernels (sequence of int): Three temporal widths.
            (default: :obj:`(85, 65, 45)`)
        pool_size (int): Temporal average-pool width. (default: :obj:`44`)
        dropout (float): Convolution dropout. (default: :obj:`0.5`)
        heads (int): Attention heads; divide three times num_filters.
            (default: :obj:`8`)
        depth (int): Post-norm Transformer blocks. (default: :obj:`5`)
        mlp_ratio (int): Feed-forward expansion. (default: :obj:`4`)
        attn_dropout (float): Attention/MLP/residual dropout. (default: :obj:`0.5`)
        pos_dropout (float): Positional dropout. (default: :obj:`0.1`)
        final_dropout (float): Readout dropout. (default: :obj:`0.25`)
        attention_scale (float, optional): Attention score multiplier. Defaults
            to :obj:`(3 * num_filters) ** -0.5`, preserving the author scaling.

    .. code-block:: python

        model = MSCFormer(num_electrodes=22, chunk_size=1000, num_classes=4)
        logits = model(torch.randn(2, 22, 1000))
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 1000,
        num_classes: int = 4,
        num_filters: int = 16,
        temporal_kernels: Sequence[int] = (85, 65, 45),
        pool_size: int = 44,
        dropout: float = 0.5,
        heads: int = 8,
        depth: int = 5,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.5,
        pos_dropout: float = 0.1,
        final_dropout: float = 0.25,
        attention_scale=None,
    ):
        super().__init__()
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('num_filters', num_filters),
            ('pool_size', pool_size),
            ('heads', heads),
            ('depth', depth),
            ('mlp_ratio', mlp_ratio),
        ):
            _positive_int(name, value)
        for name, value in (
            ('dropout', dropout),
            ('attn_dropout', attn_dropout),
            ('pos_dropout', pos_dropout),
            ('final_dropout', final_dropout),
        ):
            _probability(name, value)
        if (
            not isinstance(temporal_kernels, (list, tuple))
            or len(temporal_kernels) != 3
        ):
            raise ValueError('temporal_kernels must contain all three branch widths.')
        for kernel in temporal_kernels:
            _positive_int('temporal_kernel', kernel)
        width = num_filters * 3
        if width % heads:
            raise ValueError('heads must divide three times num_filters.')
        if chunk_size < pool_size:
            raise ValueError('chunk_size must contain at least one pooled patch.')
        if attention_scale is not None and (
            isinstance(attention_scale, bool)
            or not isinstance(attention_scale, (int, float))
            or not math.isfinite(attention_scale)
            or attention_scale <= 0
        ):
            raise ValueError('attention_scale must be finite and positive.')
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.hid_channels = width
        self.cnn = _MSCPatchEmbedding(
            num_electrodes, num_filters, temporal_kernels, pool_size, dropout
        )
        self.position = _MSCPosition(
            max(100, chunk_size // pool_size + 1), width, pos_dropout
        )
        scale = width**-0.5 if attention_scale is None else attention_scale
        self.trans = nn.Sequential(
            OrderedDict(
                layers=nn.Sequential(
                    *[
                        _MSCTransformerBlock(
                            width, heads, mlp_ratio, attn_dropout, scale
                        )
                        for _ in range(depth)
                    ]
                )
            )
        )
        self.final_layer = nn.Sequential(
            nn.Dropout(final_dropout), nn.Linear(width, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        r"""Return logits, or complete class-token features and logits."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1:] != (self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,num_electrodes,chunk_size] input.')
        tokens = self.cnn(x)
        cls_token = tokens.new_zeros((tokens.shape[0], 1, tokens.shape[2]))
        tokens = torch.cat((cls_token, tokens), dim=1) * math.sqrt(self.hid_channels)
        features = self.trans(self.position(tokens))[:, 0, :]
        logits = self.final_layer(features)
        return (features, logits) if return_features else logits
