# Adapted from Braindecode's complete MIT-licensed TMSA-Net.
# Authors: Qian Zhao and Weina Zhu. The feed-forward helper also retains
# Braindecode's BSD-3-Clause notice (Bruno Aristimunha);
# see LICENSES/TMSANet*.txt.

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from ._utils import _positive_int, _probability


class _TMSAAttention(nn.Module):
    def __init__(self, width, heads, dropout, attn_dropout):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        inner = self.head_dim * heads
        self.to_channels = Rearrange('b n d -> b d n')
        self.local_convs = nn.ModuleList(
            [nn.Conv1d(width, width, kernel, padding=kernel // 2) for kernel in (3, 5)]
        )
        self.local_bn = nn.BatchNorm1d(2 * width)
        self.local_drop = nn.Dropout(dropout)
        self.to_tokens = Rearrange('b d n -> b n d')
        self.w_q = nn.Linear(width, inner)
        self.w_k_local = nn.Linear(2 * width, inner)
        self.w_k_global = nn.Linear(width, inner)
        self.w_v = nn.Linear(width, inner)
        self.w_o = nn.Linear(inner, width)
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.att_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        local = self.to_channels(x)
        local = torch.cat([conv(local) for conv in self.local_convs], dim=1)
        local = self.to_tokens(self.local_drop(self.local_bn(local)))
        queries = self.split_heads(self.w_q(x))
        local_keys = self.split_heads(self.w_k_local(local))
        global_keys = self.split_heads(self.w_k_global(x))
        values = self.split_heads(self.w_v(x))
        dropout = self.att_drop.p if self.training else 0.0

        def attention(keys):
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            weights = scores.softmax(-1)
            if dropout > 0:
                weights = F.dropout(weights, dropout, True)
            return torch.matmul(weights, values)

        out = attention(local_keys) + attention(global_keys)
        return self.w_o(self.merge_heads(out))


class _TMSABlock(nn.Module):
    def __init__(self, width, heads, expansion, dropout, attn_dropout, fc_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = _TMSAAttention(width, heads, dropout, attn_dropout)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(expansion * width, width),
            nn.Dropout(fc_dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TMSANet(nn.Module):
    r"""Complete temporal multi-scale attention network for motor imagery.

    Zhao and Zhu. TMSA-Net: A novel attention mechanism for improved motor
    imagery EEG signal processing. Biomedical Signal Processing and Control,
    2025. `Author code <https://github.com/Whit3Zhao/TMSA-Net>`_.

    Retains both temporal convolutions, dense spatial projection, local-key
    and global-key attention, all normalization/dropout, pre-norm residual
    blocks and embedding-major dense readout. Floating inputs are
    :obj:`[batch, electrodes, time]` or :obj:`[batch, 1, electrodes, time]`.
    The caller supplies sampling rate, channel order, units and preprocessing.
    Output is logits for cross entropy; no pretrained weights are downloaded.

    Args:
        num_electrodes (int): Ordered input rows. (default: :obj:`22`)
        chunk_size (int): Time samples. (default: :obj:`1000`)
        num_classes (int): Output classes. (default: :obj:`4`)
        hid_channels (int): Convolution/Transformer width. (default: :obj:`19`)
        pool_size (int): Temporal averaging width. (default: :obj:`50`)
        pool_stride (int): Temporal averaging stride. (default: :obj:`15`)
        heads (int): Attention heads. Native projection width is
            :obj:`heads * (hid_channels // heads)`; exact divisibility is not
            required. (default: :obj:`4`)
        depth (int): Complete attention/MLP blocks. (default: :obj:`1`)
        mlp_ratio (int): Feed-forward expansion. (default: :obj:`2`)
        dropout (float): CNN/local-key dropout. (default: :obj:`0.5`)
        attn_dropout (float): Each attention-weights dropout. (default: :obj:`0.5`)
        fc_dropout (float): Both MLP dropout layers. (default: :obj:`0.5`)

    .. code-block:: python

        model = TMSANet(num_electrodes=22, chunk_size=1000, num_classes=4)
        logits = model(torch.randn(2, 22, 1000))
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 1000,
        num_classes: int = 4,
        hid_channels: int = 19,
        pool_size: int = 50,
        pool_stride: int = 15,
        heads: int = 4,
        depth: int = 1,
        mlp_ratio: int = 2,
        dropout: float = 0.5,
        attn_dropout: float = 0.5,
        fc_dropout: float = 0.5,
    ):
        super().__init__()
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('hid_channels', hid_channels),
            ('pool_size', pool_size),
            ('pool_stride', pool_stride),
            ('heads', heads),
            ('depth', depth),
            ('mlp_ratio', mlp_ratio),
        ):
            _positive_int(name, value)
        for name, value in (
            ('dropout', dropout),
            ('attn_dropout', attn_dropout),
            ('fc_dropout', fc_dropout),
        ):
            _probability(name, value)
        if hid_channels < heads:
            raise ValueError('hid_channels must be at least heads.')
        if chunk_size < pool_size:
            raise ValueError('chunk_size must contain at least one pooled token.')
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.add_channel_dim = Rearrange('b c t -> b 1 c t')
        self.temp_conv1 = nn.Conv2d(1, hid_channels, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, hid_channels, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.spatial_conv = nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1))
        self.activation = nn.GELU()
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.to_sequence = Rearrange('b d 1 t -> b d t')
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            Rearrange('b d n -> b n d'),
            *[
                _TMSABlock(
                    hid_channels, heads, mlp_ratio, dropout, attn_dropout, fc_dropout
                )
                for _ in range(depth)
            ],
        )
        self.num_tokens = (chunk_size - pool_size) // pool_stride + 1
        self.flatten = Rearrange('b n d -> b (d n)')
        self.final_layer = nn.Linear(hid_channels * self.num_tokens, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        r"""Return logits, or full embedding-major features and logits."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1:] != (self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,num_electrodes,chunk_size] input.')
        x = self.add_channel_dim(x)
        x = self.bn1(self.temp_conv1(x) + self.temp_conv2(x))
        x = self.to_sequence(self.bn2(self.activation(self.spatial_conv(x))))
        x = self.transformer(self.dropout(self.avg_pool(x)))
        features = self.flatten(x)
        logits = self.final_layer(features)
        return (features, logits) if return_features else logits
