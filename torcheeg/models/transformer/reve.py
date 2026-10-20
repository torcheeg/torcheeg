"""REVE pretrained encoders; adapted from BRAIN's MIT-licensed implementation.

Copyright (c) 2026 BRAIN - BRoader Artificial INtelligence.
Source: https://github.com/elouayas/reve_eeg
See LICENSES/REVE.txt. Official checkpoints have a separate responsible-use license.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ._utils import _positive_int, _probability


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        values = x.float()
        return (values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class _GEGLU(nn.Module):
    def forward(self, x):
        values, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * values


class _FeedForward(nn.Module):
    def __init__(self, width, hidden, use_geglu):
        super().__init__()
        self.net = nn.Sequential(_RMSNorm(width),
                                 nn.Linear(width, hidden * 2 if use_geglu else hidden, bias=False),
                                 _GEGLU() if use_geglu else nn.GELU(),
                                 nn.Linear(hidden, width, bias=False))

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, width, heads, head_width):
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        self.norm = _RMSNorm(width)
        self.to_qkv = nn.Linear(width, 3 * heads * head_width, bias=False)
        self.to_out = nn.Linear(heads * head_width, width, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        projected = self.to_qkv(self.norm(x))
        q, k, v = [part.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for part in projected.chunk(3, dim=-1)]
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(q, k, v)
        else:
            # Native classical attention for supported PyTorch versions without SDPA.
            attention = ((q @ k.transpose(-1, -2)) * (self.head_width ** -.5)).softmax(dim=-1)
            output = attention @ v
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.to_out(output)


class _Transformer(nn.Module):
    def __init__(self, width, depth, heads, head_width, hidden, use_geglu):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([_Attention(width, heads, head_width), _FeedForward(width, hidden, use_geglu)])
            for _ in range(depth)])

    def forward(self, x, return_layers):
        outputs = [x] if return_layers else None
        for attention, feedforward in self.layers:
            x = x + attention(x)
            x = x + feedforward(x)
            if return_layers:
                outputs.append(x)
        return outputs if return_layers else x


class _FourierEmbedding(nn.Module):
    def __init__(self, width, frequencies):
        super().__init__()
        self.width = width
        self.frequencies = frequencies

    def forward(self, positions):
        positions = positions.clone()
        positions[..., -1] *= .1
        frequencies = torch.arange(self.frequencies, device=positions.device, dtype=positions.dtype)
        w = frequencies
        z = w[:, None]
        y = z[:, None]
        x = y[:, None]
        positions = (positions + .4)[..., None, None, None, None, :]
        phase = (positions[..., 0] * (2 * math.pi * x / 1.8) +
                 positions[..., 1] * (2 * math.pi * y / 1.8) +
                 positions[..., 2] * (2 * math.pi * z / 1.8) +
                 positions[..., 3] * (2 * math.pi * w / 1.8))
        phase = phase.flatten(-4)[..., :self.width // 2]
        return torch.cat([phase.cos(), phase.sin()], dim=-1)


class REVE(nn.Module):
    r'''REVE: A Foundation Model for EEG -- Adapting to Any Setup with
    Large-Scale Pretraining on 25,000 Subjects.

    Input EEG is [batch, electrodes, samples], sampled at 200 Hz for the
    released weights. Supply matching [electrodes,3] or [batch,electrodes,3]
    coordinates in the author's coordinate convention. No montage, scaling,
    filtering, normalization or padding is inferred. Complete patches use
    stride patch_size-patch_overlap; a trailing incomplete patch is unused.
    Return [batch,electrodes,patches,hid_channels] features. The official
    pretrained query is retained and available through attention_pooling.

    Args:
        hid_channels (int): Embedding width. (default: :obj:`512`)
        depth (int): Transformer layers. (default: :obj:`22`)
        num_heads (int): Attention heads. (default: :obj:`8`)
        head_channels (int): Width per attention head. (default: :obj:`64`)
        mlp_ratio (float): Feedforward expansion. (default: :obj:`2.66`)
        use_geglu (bool): Use GEGLU feedforwards. (default: :obj:`True`)
        num_frequencies (int): Frequencies along each of four axes. (default: :obj:`4`)
        patch_size (int): Samples per patch. (default: :obj:`200`)
        patch_overlap (int): Overlap between consecutive patches. (default: :obj:`20`)
    '''
    variants = {
        'base': dict(hid_channels=512, depth=22, num_heads=8, head_channels=64, num_frequencies=4),
        'large': dict(hid_channels=1216, depth=22, num_heads=19, head_channels=64, num_frequencies=5),
    }

    def __init__(self, hid_channels=512, depth=22, num_heads=8, head_channels=64,
                 mlp_ratio=2.66, use_geglu=True, num_frequencies=4, patch_size=200, patch_overlap=20):
        super().__init__()
        for name, value in [('hid_channels', hid_channels), ('depth', depth), ('num_heads', num_heads),
                            ('head_channels', head_channels), ('num_frequencies', num_frequencies),
                            ('patch_size', patch_size)]:
            _positive_int(name, value)
        if (isinstance(patch_overlap, bool) or not isinstance(patch_overlap, int) or
                not 0 <= patch_overlap < patch_size):
            raise ValueError('patch_overlap must be an integer in [0,patch_size).')
        if (isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)) or
                not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hid_channels * mlp_ratio) < 1):
            raise ValueError('mlp_ratio must give a positive finite hidden width.')
        if not isinstance(use_geglu, bool) or hid_channels % 2 or 2 * num_frequencies ** 4 < hid_channels:
            raise ValueError('use_geglu must be boolean; embedding width must be even and fit the frequency grid.')
        self.hid_channels = hid_channels
        self.depth = depth
        self.num_heads = num_heads
        self.head_channels = head_channels
        self.mlp_ratio = mlp_ratio
        self.use_geglu = use_geglu
        self.num_frequencies = num_frequencies
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.feature_dim = hid_channels
        self.transformer = _Transformer(hid_channels, depth, num_heads, head_channels,
                                        int(hid_channels * mlp_ratio), use_geglu)
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_size, hid_channels))
        self.fourier4d = _FourierEmbedding(hid_channels, num_frequencies)
        self.mlp4d = nn.Sequential(nn.Linear(4, hid_channels, bias=False), nn.GELU(), nn.LayerNorm(hid_channels))
        self.ln = nn.LayerNorm(hid_channels)
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, hid_channels))

    def forward_features(self, x, positions, return_layers=False):
        r'''Return flattened [batch,electrodes*patches,width] tokens.

        return_layers=True returns the embedded input and every layer output.
        Like the released Hugging Face model, coordinates are never modified
        or jittered, and EEG is converted to float32. The native training
        repository's in-place NumPy coordinate augmentation is not applied.
        '''
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                min(x.shape) < 1 or x.shape[-1] < self.patch_size):
            raise ValueError('EEG must be nonempty floating [batch,electrodes,samples], at least one patch long.')
        if not isinstance(return_layers, bool):
            raise ValueError('return_layers must be boolean.')
        if not isinstance(positions, torch.Tensor) or not positions.is_floating_point():
            raise ValueError('positions must be floating coordinates.')
        if positions.shape == (x.shape[1], 3):
            positions = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        if positions.shape != (x.shape[0], x.shape[1], 3):
            raise ValueError('positions must be [electrodes,3] or [batch,electrodes,3].')
        x = x.float()
        positions = positions.to(x)
        patches = x.unfold(2, self.patch_size, self.patch_size - self.patch_overlap)
        batch, channels, count, _ = patches.shape
        spatial = positions.unsqueeze(2).expand(-1, -1, count, -1)
        time = torch.arange(count, device=x.device, dtype=x.dtype).reshape(1, 1, count, 1).expand(batch, channels, -1, -1)
        coordinates = torch.cat([spatial, time], dim=-1).reshape(batch, channels * count, 4)
        positional = self.ln(self.fourier4d(coordinates) + self.mlp4d(coordinates))
        tokens = self.to_patch_embedding(patches).reshape(batch, channels * count, self.hid_channels) + positional
        return self.transformer(tokens, return_layers)

    def forward(self, x, positions):
        r'''Return [batch,electrodes,patches,width] native features.'''
        tokens = self.forward_features(x, positions)
        return tokens.reshape(x.shape[0], x.shape[1], -1, self.hid_channels)

    def attention_pooling(self, features):
        r'''Pool native four-dimensional features with the pretrained query.'''
        if (not isinstance(features, torch.Tensor) or features.ndim != 4 or
                min(features.shape) < 1 or features.shape[-1] != self.hid_channels):
            raise ValueError('features must be nonempty [batch,electrodes,patches,width].')
        tokens = features.flatten(1, 2)
        scores = self.cls_query_token.expand(tokens.shape[0], -1, -1) @ tokens.transpose(1, 2)
        return (scores.div(self.hid_channels ** .5).softmax(-1) @ tokens).squeeze(1)

    @classmethod
    def from_pretrained(cls, checkpoint_path, variant='base', prefix=''):
        r'''Load a complete local official safetensors checkpoint strictly on CPU.

        No download or remote-code execution occurs. All keys, shapes, dtypes
        and finite values are required, including the pretrained query token.
        Official weights have a separate REVE Responsible Use License v1.0;
        the MIT source license does not replace those terms.
        '''
        if not isinstance(variant, str) or variant not in cls.variants or not isinstance(prefix, str):
            raise ValueError('Select a known REVE variant and a string prefix.')
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError('REVE weights require safetensors; install torcheeg[foundation].') from error
        state = load_file(str(Path(checkpoint_path)), device='cpu')
        selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        model = cls(**cls.variants[variant])
        expected = model.state_dict()
        if set(selected) != set(expected):
            raise ValueError(f'Incomplete REVE state: missing={sorted(set(expected)-set(selected))}, '
                             f'extra={sorted(set(selected)-set(expected))}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(selected, strict=True)
        return model


class REVEClassifier(nn.Module):
    r'''REVE pretrained-query pooling and a new RMSNorm/dropout/linear head.

    Return [batch,num_classes] logits. Optional fixed coordinates are stored
    as a buffer for ClassifierTrainer; per-call coordinates override them.
    The encoder and pretrained query remain frozen when freeze_backbone=True.

    Args:
        backbone (REVE): Complete REVE backbone.
        num_classes (int): Number of output classes. (default: :obj:`2`)
        dropout (float): Head dropout probability. (default: :obj:`0.1`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
        positions (torch.Tensor, optional): Fixed [electrodes,3] coordinates.
    '''
    def __init__(self, backbone, num_classes=2, dropout=.1, freeze_backbone=False, positions=None):
        super().__init__()
        if not isinstance(backbone, REVE) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a REVE backbone and boolean freeze_backbone.')
        _positive_int('num_classes', num_classes)
        _probability('dropout', dropout)
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Sequential(_RMSNorm(backbone.hid_channels), nn.Dropout(dropout),
                                        nn.Linear(backbone.hid_channels, num_classes)).to(backbone.cls_query_token)
        if positions is not None:
            if (not isinstance(positions, torch.Tensor) or not positions.is_floating_point() or
                    positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] != 3 or
                    not torch.isfinite(positions).all()):
                raise ValueError('Fixed positions must be finite floating [electrodes,3] coordinates.')
            positions = positions.detach().clone().to(backbone.cls_query_token)
        self.register_buffer('positions', positions)
        if freeze_backbone:
            self.backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, positions=None):
        if positions is None:
            positions = self.positions
        return self.classifier(self.backbone.attention_pooling(self.backbone(x, positions)))
