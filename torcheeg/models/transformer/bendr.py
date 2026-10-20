# Adapted from SPOClab-ca/DN3 under BSD-3-Clause; see LICENSES/BENDR.txt.
# Copyright (c) 2020, SPOClab-ca.
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability


class _Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class _ContextLayer(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout)
        self.linear1 = nn.Linear(width, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Native BENDR removes both TransformerEncoderLayer norms for T-fixup.
        # Express that path directly without replacing framework norm modules
        # or depending on private normalization fastpaths.
        attended = self.self_attn(x, x, x, need_weights=False)[0]
        x = x + self.dropout1(attended)
        return x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class BENDREncoder(nn.Module):
    r'''BENDR convolutional representation encoder from DN3.

    - Paper: Kostas D, Aroca-Ouellette S, Rudzicz F. BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data. Frontiers in Human Neuroscience, 2021.
    - Related Project: https://github.com/SPOClab-ca/BENDR

    Input [batch, electrodes, samples] returns [batch, hid_channels, frames].
    Six odd-width convolutions downsample by 96 with native channel dropout,
    GroupNorm and GELU. frames = ceil(samples/96). Official weights use
    hid_channels=512 and 20 inputs: 19 electrodes plus a scale channel, at
    256 Hz. The 20th channel is not another EEG electrode. See the BENDR guide
    for order, min/max normalization and scale semantics. This module does
    not resample or construct that representation. It is the complete native
    convolutional encoder, not the contextual transformer.

    Args:
        num_electrodes (int): Number of input channels, including scale. (default: :obj:`20`)
        hid_channels (int): Positive even convolution width. (default: :obj:`512`)
        dropout (float): Channel dropout after convolutions. (default: :obj:`0.0`)
    '''
    def __init__(self, num_electrodes=20, hid_channels=512, dropout=0.):
        super().__init__()
        _positive_int('num_electrodes', num_electrodes)
        _positive_int('hid_channels', hid_channels)
        _probability('dropout', dropout)
        if hid_channels % 2:
            raise ValueError('hid_channels must be even.')
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.encoder = nn.Sequential()
        for layer, stride in enumerate((3, 2, 2, 2, 2, 2)):
            self.encoder.add_module(f'Encoder_{layer}', nn.Sequential(
                nn.Conv1d(num_electrodes if layer == 0 else hid_channels,
                          hid_channels, 3, stride=stride, padding=1),
                nn.Dropout1d(dropout), nn.GroupNorm(hid_channels // 2, hid_channels), nn.GELU()))

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                min(x.shape) < 1 or x.shape[1] != self.num_electrodes):
            raise ValueError('Input must be floating [batch, num_electrodes, samples] with nonempty axes.')
        return self.encoder(x)


class BENDRContextualizer(nn.Module):
    r'''Native BENDR transformer contextualizer with explicit masks.

    Input [batch, hid_channels, frames] returns [batch, hid_channels, frames+1].
    The prepended native constant start token produces output position zero.
    Grouped weight-normalized relative position convolution, input conditioning,
    eight norm-free T-fixup transformer blocks and output projection retain
    official checkpoint names. The source's unused final norm is also retained.
    Use explicit mask_t [batch, frames] and mask_c [batch, hid_channels] boolean
    tensors for time/channel masking; masks never modify caller tensors.
    Automatic random pretraining masks and contrastive loss are not implied.

    Args:
        hid_channels (int): Encoder feature width, divisible by 16. (default: :obj:`512`)
        mlp_channels (int): Transformer feed-forward width. (default: :obj:`3076`)
        num_heads (int): Attention heads; must divide 3*hid_channels. (default: :obj:`8`)
        depth (int): Transformer blocks. (default: :obj:`8`)
        dropout (float): Conditioning, attention and feed-forward dropout. (default: :obj:`0.15`)
        layer_drop (float): Probability of skipping a block while training. (default: :obj:`0.0`)
        start_token (float): Constant prepended before contextualization. (default: :obj:`-5.0`)
    '''
    def __init__(self, hid_channels=512, mlp_channels=3076, num_heads=8, depth=8,
                 dropout=.15, layer_drop=0., start_token=-5.):
        super().__init__()
        for name, value in [('hid_channels', hid_channels), ('mlp_channels', mlp_channels),
                            ('num_heads', num_heads), ('depth', depth)]:
            _positive_int(name, value)
        _probability('dropout', dropout)
        _probability('layer_drop', layer_drop)
        if hid_channels % 16 or (3 * hid_channels) % num_heads:
            raise ValueError('hid_channels must be divisible by 16 and 3*hid_channels by num_heads.')
        if isinstance(start_token, bool) or not isinstance(start_token, (int, float)) or not math.isfinite(start_token):
            raise ValueError('start_token must be a finite number.')
        self.hid_channels = hid_channels
        self.layer_drop = layer_drop
        self.start_token = start_token
        width = 3 * hid_channels
        self.norm = nn.LayerNorm(width)
        self.transformer_layers = nn.ModuleList([
            _ContextLayer(width, num_heads, mlp_channels, dropout) for _ in range(depth)])
        self.mask_replacement = nn.Parameter(torch.normal(0, hid_channels ** -.5, size=(hid_channels,)))
        conv = nn.Conv1d(hid_channels, hid_channels, 25, padding=12, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / width)
        nn.init.zeros_(conv.bias)
        # Keep the native weight_g/weight_v state layout, including dim=2.
        self.relative_position = nn.Sequential(nn.utils.weight_norm(conv, dim=2), nn.GELU())
        self.input_conditioning = nn.Sequential(_Permute((0, 2, 1)), nn.LayerNorm(hid_channels),
            nn.Dropout(dropout), _Permute((0, 2, 1)), nn.Conv1d(hid_channels, width, 1), _Permute((2, 0, 1)))
        self.output_layer = nn.Conv1d(width, hid_channels, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                with torch.no_grad():
                    module.weight.mul_(.67 * depth ** -.25)

    @staticmethod
    def _mask(mask, shape, name, device):
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape):
            raise ValueError(f'{name} must be a boolean tensor with shape {tuple(shape)}.')
        return mask.to(device=device)

    def forward(self, x, mask_t=None, mask_c=None):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                min(x.shape) < 1 or x.shape[1] != self.hid_channels):
            raise ValueError('Input must be floating [batch, hid_channels, frames] with nonempty axes.')
        mask_t = self._mask(mask_t, (x.shape[0], x.shape[2]), 'mask_t', x.device)
        mask_c = self._mask(mask_c, x.shape[:2], 'mask_c', x.device)
        if mask_t is not None:
            x = torch.where(mask_t[:, None], self.mask_replacement[None, :, None], x)
        if mask_c is not None:
            x = torch.where(mask_c[:, :, None], torch.zeros_like(x), x)
        x = x + self.relative_position(x)
        x = self.input_conditioning(x)
        x = torch.cat((x.new_full((1, x.shape[1], x.shape[2]), self.start_token), x), dim=0)
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))


def _load_native_state(module, path, prefix):
    if not isinstance(prefix, str):
        raise ValueError('Checkpoint prefix must be a string.')
    checkpoint = torch.load(Path(path), weights_only=True, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a native tensor dictionary.')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError('Checkpoint keys must be strings.')
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    expected = module.state_dict()
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f'Incomplete BENDR state: missing={sorted(missing)}, extra={sorted(extra)}.')
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
        if not torch.isfinite(value).all():
            raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
    module.load_state_dict(state, strict=True)


class BENDR(nn.Module):
    r'''Complete BENDR convolutional encoder and contextual transformer.

    Input [batch, num_electrodes, samples] returns the contextualized start
    token [batch, hid_channels]. forward_features returns all contextualized
    positions [batch, hid_channels, ceil(samples/96)+1]. Use encoder(x) for
    convolutional features only. Official weights require 19 ordered scalp
    electrodes plus an explicit scale channel, at 256 Hz; see the user guide.
    Missing electrodes must follow a declared mapping, never arbitrary indices.
    This backbone does not implement the full contrastive pretraining objective.

    .. automethod:: forward_features
    .. automethod:: from_pretrained

    Args:
        num_electrodes (int): Input channels including scale. (default: :obj:`20`)
        hid_channels (int): Convolution/context output width. (default: :obj:`512`)
        mlp_channels (int): Context feed-forward width. (default: :obj:`3076`)
        num_heads (int): Context attention heads. (default: :obj:`8`)
        depth (int): Context transformer blocks. (default: :obj:`8`)
        encoder_dropout (float): Channel dropout in the encoder. (default: :obj:`0.0`)
        dropout (float): Context dropout. (default: :obj:`0.15`)
        layer_drop (float): Context block skip probability. (default: :obj:`0.0`)
    '''
    def __init__(self, num_electrodes=20, hid_channels=512, mlp_channels=3076,
                 num_heads=8, depth=8, encoder_dropout=0., dropout=.15, layer_drop=0.):
        super().__init__()
        self.hid_channels = hid_channels
        self.encoder = BENDREncoder(num_electrodes, hid_channels, encoder_dropout)
        self.contextualizer = BENDRContextualizer(hid_channels, mlp_channels, num_heads, depth, dropout, layer_drop)

    def forward_features(self, x, mask_t=None, mask_c=None):
        r'''Return all contextualized positions, including the first start token.'''
        return self.contextualizer(self.encoder(x), mask_t=mask_t, mask_c=mask_c)

    def forward(self, x, mask_t=None, mask_c=None):
        return self.forward_features(x, mask_t=mask_t, mask_c=mask_c)[:, :, 0]

    @classmethod
    def from_pretrained(cls, encoder_path, contextualizer_path, encoder_prefix='', contextualizer_prefix=''):
        r'''Strictly load both canonical official local weight files safely.

        Use encoder.pt and contextualizer.pt from the author's v0.1-alpha
        release. Validate every tensor, including native unused norm state;
        reject incomplete/foreign states, shape/type mismatches and nonfinite
        values. Deserialize with weights_only=True and no unsafe fallback.
        Optional prefixes explicitly select each component. No downloads occur.
        The method loads canonical 20-input, 512-wide, eight-block weights;
        verify official file checksums and construct the corresponding inputs.
        '''
        model = cls()
        _load_native_state(model.encoder, encoder_path, encoder_prefix)
        _load_native_state(model.contextualizer, contextualizer_path, contextualizer_prefix)
        return model


class BENDRClassifier(nn.Module):
    r'''Linear classification of the complete BENDR contextualized start token.

    Return [batch, num_classes] logits. The head is newly initialized. Freezing
    the backbone keeps its encoder/contextualizer in evaluation mode even when
    the head trains. Pass explicit masks through forward when required.

    Args:
        backbone (BENDR): Complete convolutional/contextual backbone.
        num_classes (int): Number of output classes. (default: :obj:`2`)
        freeze_backbone (bool): Train only the head. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, BENDR):
            raise ValueError('backbone must be a BENDR model.')
        _positive_int('num_classes', num_classes)
        if not isinstance(freeze_backbone, bool):
            raise ValueError('freeze_backbone must be a boolean.')
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Linear(backbone.hid_channels, num_classes).to(next(backbone.parameters()))
        if freeze_backbone:
            self.backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, mask_t=None, mask_c=None):
        return self.classifier(self.backbone(x, mask_t=mask_t, mask_c=mask_c))
