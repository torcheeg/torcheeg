# Adapted from ycq091044/BIOT and lucidrains/linear-attention-transformer.
# Copyright (c) 2023 Chaoqi Yang; Copyright (c) 2020 Phil Wang.
# MIT licenses and source revisions: LICENSES/BIOT.txt.
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int, _probability


class _LinearAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = [projection(x).reshape(batch, tokens, self.heads,
            width // self.heads).transpose(1, 2)
            for projection in (self.to_q, self.to_k, self.to_v)]
        # Native noncausal linear attention, not softmax(QK^T)V.
        q = q.softmax(dim=-1) * (width // self.heads) ** -.5
        k = k.softmax(dim=-2)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        attended = torch.einsum('bhnd,bhde->bhne', q, context)
        return self.dropout(self.to_out(attended.transpose(1, 2).reshape(batch, tokens, width)))


class _FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.w2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class _PreNorm(nn.Module):
    def __init__(self, width, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.fn(self.norm(x))


class _Chunk(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fn = _FeedForward(width)

    def forward(self, x):
        return self.fn(x)


class _Sequence(nn.Module):
    def __init__(self, width, heads, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([
            _PreNorm(width, _LinearAttention(width, heads, dropout)),
            _PreNorm(width, _Chunk(width))]) for _ in range(depth)])

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = x + attention(x)
            x = x + feed_forward(x)
        return x


class _Transformer(nn.Module):
    def __init__(self, width, heads, depth, dropout):
        super().__init__()
        self.layers = _Sequence(width, heads, depth, dropout)

    def forward(self, x):
        return self.layers(x)


class _PatchEmbedding(nn.Module):
    def __init__(self, n_fft, width):
        super().__init__()
        self.projection = nn.Linear(n_fft // 2 + 1, width)

    def forward(self, x):
        return self.projection(x.transpose(-1, -2))


class _PositionalEncoding(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        position = torch.arange(1000).unsqueeze(1).float()
        frequency = torch.exp(torch.arange(0, width, 2).float() * -(math.log(10000.) / width))
        pe = torch.zeros(1000, width)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1]])


class BIOT(nn.Module):
    r'''BIOT pretrained biosignal encoder with native linear attention.

    - Paper: Yang C, Westover MB, Sun J. BIOT: Biosignal Transformer for Cross-data Learning in the Wild. NeurIPS, 2023.
    - Related Project: https://github.com/ycq091044/BIOT

    Input is a real floating [batch, electrodes, samples] tensor. Return one
    [batch, hid_channels] mean-pooled representation. STFT uses a rectangular
    window, no centering, and magnitude without logarithm or normalization.
    Each channel starts its sinusoidal positions at zero; channel-major tokens
    then enter noncausal linear attention. No extra dependency is required.

    Official weights use 200 Hz, width 256, eight heads, four blocks, n_fft=200
    and hop_length=100. The PREST variant has 16 bipolar channel tokens; the
    other two have 18. Supply signals in the corresponding montage and units,
    or explicitly select channel_ids for a subset/reordering. This class does
    not infer electrodes, convert units, resample or perform artifact removal.
    Each channel can contain at most 1000 STFT frames. Native global linear
    attention does not impose a 1024-token limit on their concatenation.

    .. code-block:: python

        from torcheeg.models import BIOT
        model = BIOT.from_pretrained('./EEG-PREST-16-channels.ckpt', variant='prest-16ch')
        features = model(eeg)  # [batch, 16, samples] -> [batch, 256]

    .. automethod:: forward_features
    .. automethod:: from_pretrained

    Args:
        num_electrodes (int): Number of channel tokens. (default: :obj:`18`)
        hid_channels (int): Even embedding width divisible by num_heads. (default: :obj:`256`)
        num_heads (int): Number of linear attention heads. (default: :obj:`8`)
        depth (int): Number of transformer blocks. (default: :obj:`4`)
        n_fft (int): Samples in the rectangular STFT window. (default: :obj:`200`)
        hop_length (int): STFT frame stride. (default: :obj:`100`)
        dropout (float): Positional encoding dropout. (default: :obj:`0.1`)
        attn_dropout (float): Dropout after attention output projection. (default: :obj:`0.2`)
    '''
    variants = {'prest-16ch': 16, 'shhs-prest-18ch': 18, 'six-datasets-18ch': 18}

    def __init__(self, num_electrodes=18, hid_channels=256, num_heads=8, depth=4,
                 n_fft=200, hop_length=100, dropout=.1, attn_dropout=.2):
        super().__init__()
        for name, value in [('num_electrodes', num_electrodes), ('hid_channels', hid_channels),
                            ('num_heads', num_heads), ('depth', depth), ('n_fft', n_fft),
                            ('hop_length', hop_length)]:
            _positive_int(name, value)
        _probability('dropout', dropout)
        _probability('attn_dropout', attn_dropout)
        if hid_channels % 2 or hid_channels % num_heads:
            raise ValueError('hid_channels must be even and divisible by num_heads.')
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_embedding = _PatchEmbedding(n_fft, hid_channels)
        self.transformer = _Transformer(hid_channels, num_heads, depth, attn_dropout)
        self.positional_encoding = _PositionalEncoding(hid_channels, dropout)
        self.channel_tokens = nn.Embedding(num_electrodes, hid_channels)
        self.register_buffer('index', torch.arange(num_electrodes))

    def _channel_ids(self, x, channel_ids):
        if channel_ids is None:
            if x.shape[1] > self.num_electrodes:
                raise ValueError('Input has more electrodes than available channel tokens.')
            return self.index[:x.shape[1]]
        if not isinstance(channel_ids, (tuple, list)) or len(channel_ids) != x.shape[1]:
            raise ValueError('channel_ids must be a list or tuple with one index per input electrode.')
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= self.num_electrodes for i in channel_ids):
            raise ValueError('channel_ids must contain valid integer channel token indices.')
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError('channel_ids must not contain duplicates.')
        return self.index[list(channel_ids)]

    def forward_features(self, x, channel_ids=None):
        r'''Return [batch, hid_channels] embeddings, with optional channel token IDs.

        channel_ids maps each input channel to its native pretrained token.
        Omitting it selects tokens 0 through electrodes-1. It does not transform
        unipolar signals into bipolar derivations. Dropout follows train/eval.
        '''
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                min(x.shape) < 1 or x.shape[-1] < self.n_fft):
            raise ValueError('Input must be floating [batch, electrodes, samples] with samples >= n_fft and nonempty axes.')
        frames = 1 + (x.shape[-1] - self.n_fft) // self.hop_length
        if frames > 1000:
            raise ValueError('At most 1000 STFT frames per channel are supported.')
        ids = self._channel_ids(x, channel_ids)
        embedded = []
        # Retain native per-channel token/dropout ordering, including training.
        for channel in range(x.shape[1]):
            spectrum = torch.stft(x[:, channel], n_fft=self.n_fft, hop_length=self.hop_length,
                window=x.new_ones(self.n_fft), center=False, onesided=True, return_complex=True).abs()
            tokens = self.patch_embedding(spectrum) + self.channel_tokens(ids[channel])[None, None]
            embedded.append(self.positional_encoding(tokens))
        return self.transformer(torch.cat(embedded, dim=1)).mean(dim=1)

    def forward(self, x, channel_ids=None):
        return self.forward_features(x, channel_ids=channel_ids)

    @classmethod
    def from_pretrained(cls, checkpoint_path, variant='shhs-prest-18ch', prefix=''):
        r'''Load all tensors from one of the three official local encoder checkpoints.

        Variants: prest-16ch, shhs-prest-18ch, six-datasets-18ch. Select the
        matching filename from the author repository. Downloads are explicit
        user actions; this method never accesses the network. Accept a native
        tensor dictionary or a state_dict container, with an optional explicit
        prefix. Require every key, exact shapes/dtypes, finite floating values,
        and canonical channel index/position buffers. Always use weights_only
        loading without unsafe fallback. Call eval() for deterministic inference.
        A matching architecture alone does not authenticate a file's provenance;
        verify the supplied file against the author's checkpoint checksum.
        '''
        if not isinstance(variant, str) or variant not in cls.variants:
            raise ValueError('Unknown BIOT pretrained variant.')
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model = cls(num_electrodes=cls.variants[variant])
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete BIOT state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name, value in selected.items():
            target = expected[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
            if name == 'index' and not torch.equal(value, target):
                raise ValueError('Noncanonical BIOT channel indices.')
            # Official GPU-generated sinusoidal tables differ from CPU tables
            # by up to 3.06e-5. Keep the supplied table, validate within 4e-5.
            if name == 'positional_encoding.pe' and not torch.allclose(value, target, rtol=0., atol=4e-5):
                raise ValueError('Noncanonical BIOT positional encoding.')
        model.load_state_dict(selected, strict=True)
        return model


class BIOTClassifier(nn.Module):
    r'''BIOT encoder with the author's ELU and linear classification head.

    The backbone is a :class:`BIOT` instance, optionally loaded from official
    weights. Return [batch, num_classes] logits; apply cross entropy directly.
    When freeze_backbone=True, only the head trains and backbone dropout stays
    disabled. By default the complete model trains. Channel IDs follow BIOT.
    Classification weights are newly initialized, not pretrained task weights.

    Args:
        backbone (BIOT): Encoder, with its architecture and preprocessing contract.
        num_classes (int): Number of output classes. (default: :obj:`2`)
        freeze_backbone (bool): Train only the classification head. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, BIOT):
            raise ValueError('backbone must be a BIOT encoder.')
        _positive_int('num_classes', num_classes)
        if not isinstance(freeze_backbone, bool):
            raise ValueError('freeze_backbone must be a boolean.')
        self.biot = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Sequential(nn.ELU(), nn.Linear(backbone.hid_channels, num_classes)).to(
            backbone.channel_tokens.weight)
        if freeze_backbone:
            self.biot.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.biot.eval()
        return self

    def forward(self, x, channel_ids=None):
        return self.classifier(self.biot(x, channel_ids=channel_ids))
