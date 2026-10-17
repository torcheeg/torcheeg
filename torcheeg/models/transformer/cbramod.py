# Adapted from wjq-learning/CBraMod under MIT; see LICENSES/CBraMod.txt.
# Copyright (c) 2025 Jiquan Wang.
import copy
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability


class _Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.num_heads = heads
        self.dropout = dropout
        # Retain native MultiheadAttention checkpoint names without its
        # inference fastpath or private framework implementation helpers.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(
            batch, tokens, 3, self.num_heads, width // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class _CrissCrossBlock(nn.Module):
    def __init__(self, width, heads, mlp_channels, dropout, norm_eps):
        super().__init__()
        self.self_attn_s = _Attention(width // 2, heads // 2, dropout)
        self.self_attn_t = _Attention(width // 2, heads // 2, dropout)
        self.linear1 = nn.Linear(width, mlp_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_channels, width)
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        batch, channels, patches, width = x.shape
        spatial, temporal = self.norm1(x).chunk(2, dim=-1)
        spatial = spatial.transpose(1, 2).reshape(batch * patches, channels, width // 2)
        temporal = temporal.reshape(batch * channels, patches, width // 2)
        spatial = self.self_attn_s(spatial).reshape(batch, patches, channels, width // 2).transpose(1, 2)
        temporal = self.self_attn_t(temporal).reshape(batch, channels, patches, width // 2)
        x = x + self.dropout1(torch.cat((spatial, temporal), dim=-1))
        return x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))


class _Encoder(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(block) for _ in range(depth)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class _PatchEmbedding(nn.Module):
    def __init__(self, patch_size, width, dropout, position_kernel):
        super().__init__()
        steps = (patch_size + 24) // 25
        channels = width // steps
        groups = math.gcd(5, channels)
        self.width = width
        self.frequency_bins = patch_size // 2 + 1
        self.positional_encoding = nn.Sequential(nn.Conv2d(width, width,
            kernel_size=position_kernel, padding=tuple(size // 2 for size in position_kernel), groups=width))
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, channels, (1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(groups, channels), nn.GELU(),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.GroupNorm(groups, channels), nn.GELU(),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.GroupNorm(groups, channels), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(self.frequency_bins, width), nn.Dropout(dropout))

    def forward(self, x, mask):
        batch, channels, patches, samples = x.shape
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_encoding.to(dtype=x.dtype), x)
        flat = x.reshape(batch, 1, channels * patches, samples)
        temporal = self.proj_in(flat).permute(0, 2, 1, 3).reshape(batch, channels, patches, self.width)
        # FFT lengths such as the canonical 200 samples are not supported by
        # half-precision CUDA FFT. Keep float64 inputs; promote only low precision.
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        magnitude = torch.fft.rfft(fft_input, dim=-1, norm='forward').abs()
        # Complex abs has a finite zero-input gradient, unlike sqrt(re² + im²).
        spectral = self.spectral_proj(magnitude.to(dtype=self.spectral_proj[0].weight.dtype))
        embedded = temporal + spectral
        positional = self.positional_encoding(embedded.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return embedded + positional


def _weights_init(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class CBraMod(nn.Module):
    r'''CBraMod spatial/temporal backbone with native patch reconstruction.

    - Paper: Wang J, Zhao S, Luo Z, et al. CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR, 2025.
    - URL: https://openreview.net/forum?id=NPNUHgHF2w
    - Related Project: https://github.com/wjq-learning/CBraMod

    Input is [batch, electrodes, patches, patch_size]. The temporal convolution
    and normalized real-FFT magnitude form patch features. A depthwise
    position convolution and pre-normalized criss-cross transformer mix
    spatial and temporal information through separate halves of the features.
    Electrode order and physical patch duration must remain explicit.

    forward() applies the native reconstruction projection and returns
    [batch, electrodes, patches, out_features]. forward_features() returns
    [batch, electrodes, patches, hid_channels] before that projection.
    feature(flatten=True) returns flattened backbone features. A classification
    or regression loss/head is not implied by the backbone. Reconstruction
    output alone does not implement a complete masked-pretraining objective.

    Canonical weights use 200 Hz, 200-sample one-second patches, width 200,
    12 blocks and eight heads. Architecture changes are random-model variants,
    not equivalent pretrained settings. The convolution output has
    ceil(patch_size/25) temporal steps: hid_channels must be divisible by that
    count, and by an even num_heads. No sampling-rate inference or implicit
    resampling occurs. A source preprocessing protocol must set signal units,
    reference, channel order, filters and amplitude scaling.

    .. code-block:: python

        import torch
        from torcheeg.models import CBraMod

        model = CBraMod()
        eeg = torch.randn(2, 19, 4, 200)
        reconstructed = model(eeg)
        features = model.forward_features(eeg)

    .. automethod:: forward_features
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        patch_size (int): Samples per temporal patch. (default: :obj:`200`)
        out_features (int): Reconstructed features per patch. (default: :obj:`200`)
        hid_channels (int): Transformer and patch feature width. (default: :obj:`200`)
        mlp_channels (int): Feed-forward hidden width. (default: :obj:`800`)
        depth (int): Number of criss-cross blocks. (default: :obj:`12`)
        num_heads (int): Total even head count, split across spatial/temporal attention. (default: :obj:`8`)
        dropout (float): Attention, feed-forward and spectral dropout. (default: :obj:`0.1`)
        norm_eps (float): Transformer LayerNorm epsilon. (default: :obj:`1e-5`)
        position_kernel (tuple): Odd spatial/temporal depthwise kernel sizes. (default: :obj:`(19, 7)`)
    '''

    def __init__(self, patch_size=200, out_features=200, hid_channels=200, mlp_channels=800,
                 depth=12, num_heads=8, dropout=.1, norm_eps=1e-5, position_kernel=(19, 7)):
        super().__init__()
        for name, value in [('patch_size', patch_size), ('out_features', out_features),
                            ('hid_channels', hid_channels), ('mlp_channels', mlp_channels),
                            ('depth', depth), ('num_heads', num_heads)]:
            _positive_int(name, value)
        _probability('dropout', dropout)
        if isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive.')
        if num_heads % 2 or hid_channels % num_heads or hid_channels % ((patch_size + 24) // 25):
            raise ValueError('Require even num_heads and hid_channels divisible by num_heads and ceil(patch_size/25).')
        if not isinstance(position_kernel, (tuple, list)) or len(position_kernel) != 2:
            raise ValueError('position_kernel must contain two positive odd integers.')
        for size in position_kernel:
            _positive_int('position_kernel size', size)
            if size % 2 != 1:
                raise ValueError('position_kernel sizes must be odd.')
        self.patch_size = patch_size
        self.out_features = out_features
        self.hid_channels = hid_channels
        self.mlp_channels = mlp_channels
        self.depth = depth
        self.num_heads = num_heads
        self.patch_embedding = _PatchEmbedding(patch_size, hid_channels, dropout, tuple(position_kernel))
        self.encoder = _Encoder(_CrissCrossBlock(hid_channels, num_heads, mlp_channels, dropout, norm_eps), depth)
        self.proj_out = nn.Sequential(nn.Linear(hid_channels, out_features))
        self.apply(_weights_init)

    def _check_input(self, x, mask):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 4 or
                x.shape[-1] != self.patch_size or min(x.shape[:3]) < 1):
            raise ValueError('Input must be a real floating [batch, electrodes, patches, patch_size] tensor with nonempty axes.')
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor) or mask.shape != x.shape[:3]:
            raise ValueError('mask must have shape [batch, electrodes, patches].')
        if mask.dtype != torch.bool:
            if mask.is_complex() or torch.any((mask != 0) & (mask != 1)):
                raise ValueError('mask must contain only boolean or binary values.')
        return mask.to(device=x.device, dtype=torch.bool)

    def forward_features(self, x, mask=None):
        r'''Return backbone patch features before the reconstruction projection.

        mask optionally selects patches to replace with the native fixed
        mask_encoding before both temporal and spectral embedding. Its shape
        is [batch, electrodes, patches]; True/1 means masked. Boolean masks
        allow graph capture without value-dependent binary validation.
        '''
        mask = self._check_input(x, mask)
        return self.encoder(self.patch_embedding(x, mask))

    def feature(self, x, flatten=False, mask=None):
        r'''Extract backbone features, optionally flattened from axis one.'''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        features = self.forward_features(x, mask=mask)
        return features.flatten(1) if flatten else features

    def forward(self, x, mask=None):
        return self.proj_out(self.forward_features(x, mask=mask))

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', **kwargs):
        r'''Load a complete local native CBraMod tensor state safely.

        Accept a tensor dictionary or a safely loadable state_dict checkpoint.
        Select one explicit string prefix; other outer components are ignored.
        Require all native keys including the reconstruction projection, exact
        shapes, floating tensors and finite values. Read with weights_only=True
        and no unsafe fallback. Constructor settings are explicit; the model
        downloads nothing and cannot infer preprocessing from tensor shapes.
        '''
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model = cls(**kwargs)
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete CBraMod state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name, value in selected.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != expected[name].shape:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(selected, strict=True)
        return model
