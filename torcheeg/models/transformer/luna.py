# Adapted from pulp-bio/BioFoundation, modified for TorchEEG.
# Copyright (C) 2025 ETH Zurich, Switzerland. Apache-2.0.
# MLP: Copyright 2020 Ross Wightman. Rotary: Copyright 2021 Phil Wang.
# Sources, notices and licenses: LICENSES/LUNA.txt.
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability


class _Mlp(nn.Module):
    def __init__(self, inputs, hidden, outputs, dropout=0., norm=False):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden) if norm else nn.Identity()
        self.fc2 = nn.Linear(hidden, outputs)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class _DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * (mask / keep)


class _RotaryEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.freqs = nn.Parameter(1. / (10000 ** (torch.arange(0, width, 2).float() / width)), requires_grad=False)

    def forward(self, x):
        positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).to(self.freqs.dtype)
        angles = torch.einsum('n,f->nf', positions, self.freqs).repeat_interleave(2, dim=-1)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        left, right = pairs.unbind(-1)
        rotated = torch.stack((-right, left), dim=-1).flatten(-2)
        return (x * angles.cos() + rotated * angles.sin()).to(x.dtype)


class _RotaryAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads = heads
        self.rotary_emb = _RotaryEmbedding(width // heads)
        self.qkv_proj = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv_proj(x).reshape(batch, tokens, 3, self.num_heads,
            width // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotary_emb(q), self.rotary_emb(k)
        attention = ((q @ k.transpose(-2, -1)) * (width // self.num_heads) ** -.5).softmax(-1)
        return self.proj((attention @ v).transpose(1, 2).reshape(batch, tokens, width))


class _RotaryBlock(nn.Module):
    def __init__(self, width, heads, ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = _RotaryAttention(width, heads)
        self.drop_path1 = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = _Mlp(width, int(width * ratio), width, norm=True)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class _PatchEmbedding(nn.Module):
    def __init__(self, width, patch_size):
        super().__init__()
        self.patch_size = patch_size
        hidden = width // 4
        half = patch_size // 2
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, hidden, (1, half - 1), stride=(1, half // 2), padding=(0, half // 2 - 1)),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 1)), nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 1)), nn.GroupNorm(4, hidden), nn.GELU())

    def forward(self, x):
        batch, channels, samples = x.shape
        patches = x.reshape(batch, 1, channels * (samples // self.patch_size), self.patch_size)
        # Native flatten order is within-patch time, then convolution channels.
        return self.proj_in(patches).permute(0, 2, 3, 1).flatten(-2)


class _FrequencyEmbedding(nn.Module):
    def __init__(self, width, patch_size):
        super().__init__()
        self.patch_size = patch_size
        bins = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = _Mlp(bins, 4 * bins, width)

    def forward(self, x):
        batch, channels, samples = x.shape
        patches = x.reshape(batch, channels, samples // self.patch_size, self.patch_size)
        fft_input = patches.float() if patches.dtype in (torch.float16, torch.bfloat16) else patches
        spectrum = torch.fft.rfft(fft_input, dim=-1)
        features = torch.cat((spectrum.abs(), spectrum.angle()), dim=-1).to(x.dtype)
        return self.frequency_to_embed(features).flatten(1, 2)


def _position_encoding(coords, width):
    frequencies = 2. ** torch.arange(width // 6, device=coords.device).float()
    phase = coords.unsqueeze(-1) * frequencies
    embedded = torch.stack((phase.sin(), phase.cos()), dim=-1).permute(0, 1, 3, 2, 4).flatten(2)
    return F.pad(embedded, (0, width - embedded.shape[-1]))


class _CrossAttention(nn.Module):
    def __init__(self, width, heads, queries, ratio):
        super().__init__()
        hidden = int(width * ratio)
        self.query_embed = nn.Parameter(torch.randn(1, queries, width))
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=.1, batch_first=True)
        # Keep the native checkpoint spelling of this frozen scalar.
        self.temparature = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.ffn = _Mlp(width, hidden, width, dropout=.1, norm=True)
        self.keys_norm = nn.LayerNorm(width)
        self.values_norm = nn.LayerNorm(width)
        self.queries_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, heads, hidden, activation='gelu', batch_first=True, norm_first=True)
        self.query_self_attn = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        queries = self.queries_norm(self.query_embed.expand(x.shape[0], -1, -1))
        attended, scores = self.cross_attention(queries, self.keys_norm(x), self.values_norm(x))
        return self.query_self_attn(self.ffn(attended) + attended), scores


class _ReconstructionHead(nn.Module):
    def __init__(self, width, heads, patch_size, queries):
        super().__init__()
        self.queries = queries
        self.decoder_pred = nn.TransformerDecoder(nn.TransformerDecoderLayer(width, heads,
            dim_feedforward=4 * width, dropout=0., activation='gelu', batch_first=True, norm_first=True), num_layers=1)
        self.norm = nn.LayerNorm(width)
        self.decoder_linear = _Mlp(width, 4 * width, patch_size)

    def forward(self, latent, decoder_queries):
        batch, patches, total_width = latent.shape
        memory = latent.reshape(batch * patches, self.queries, total_width // self.queries)
        decoded = self.decoder_linear(self.norm(self.decoder_pred(decoder_queries, memory)))
        return decoded.reshape(batch, patches, decoded.shape[1], -1).permute(0, 2, 1, 3).flatten(2)


class _ChannelEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embeddings = nn.Embedding(90, width)

    def forward(self, indices):
        return self.embeddings(indices)


def _initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class LUNA(nn.Module):
    r'''Complete LUNA pretrained reconstruction model and topology-aware encoder.

    - Related Project: https://github.com/pulp-bio/BioFoundation
    - Weights: https://huggingface.co/PulpBio/LUNA

    Supply floating EEG [batch, electrodes, samples] and electrode coordinates
    [batch, electrodes, 3] or [electrodes, 3]. Samples must be divisible by
    patch_size. Canonical inputs use 256 Hz, five-second windows, 0.1--75 Hz
    filtering, an explicitly selected power-line notch and per-channel z-score
    normalization. Preprocessing, reference and coordinate frame are caller
    responsibilities. Coordinates are min/max normalized per example/axis.

    forward_features returns [batch, patches, num_queries*hid_channels].
    forward reconstructs [batch, electrodes, samples] and additionally requires
    native channel IDs; get_channel_ids maps names to the fixed 90-row native
    pretraining vocabulary. Unknown electrodes can use encoder features but
    cannot silently select a learned decoder embedding. Optional boolean masks
    select patches [batch, electrodes, patches] or samples [batch, electrodes,
    samples]; any masked sample replaces its entire patch embedding before
    attention. This is native feature masking after shared convolutional
    GroupNorm, so raw masked samples can still affect visible-token statistics.
    It does not imply raw-sample isolation.

    Source reconstruction/channel-position repeats are corrected so examples
    with different layouts remain separate in one batch. Position jitter is
    opt-in via position_noise: the source's masked path uses 0.02 even during
    evaluation, whereas the default here is deterministic. Reconstruction alone
    does not define the complete masking schedule or pretraining objectives.
    Source code is Apache-2.0; official weights separately use CC-BY-ND-4.0.
    See the guide before sharing any weight artifacts.

    .. automethod:: forward_features
    .. automethod:: get_channel_ids
    .. automethod:: from_pretrained

    Args:
        patch_size (int): Samples per patch, multiple of four and at least eight. (default: :obj:`40`)
        hid_channels (int): Width per query, divisible by 16. (default: :obj:`64`)
        num_queries (int): Learned channel-unification queries. (default: :obj:`4`)
        depth (int): Temporal transformer blocks. (default: :obj:`8`)
        num_heads (int): Heads per query feature width. (default: :obj:`2`)
        mlp_ratio (float): Transformer hidden-width multiplier. (default: :obj:`4.0`)
        drop_path (float): Temporal stochastic depth. (default: :obj:`0.0`)
    '''
    variants = {'base': dict(hid_channels=64, num_queries=4, depth=8),
                'large': dict(hid_channels=96, num_queries=6, depth=10),
                'huge': dict(hid_channels=128, num_queries=8, depth=24)}
    channel_names = (
        'A1-T3', 'AF3', 'AF4', 'C1', 'C2', 'C3',
        'C3-CZ', 'C3-P3', 'C4', 'C4-P4', 'C4-T4', 'C5',
        'C6', 'CB1', 'CB2', 'CP1', 'CP2', 'CP3',
        'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'CZ-C4',
        'F1', 'F10', 'F2', 'F3', 'F3-C3', 'F4',
        'F4-C4', 'F5', 'F6', 'F7', 'F7-T3', 'F8',
        'F8-T4', 'F9', 'FC1', 'FC2', 'FC3', 'FC4',
        'FC5', 'FC6', 'FCZ', 'FP1', 'FP1-F3', 'FP1-F7',
        'FP2', 'FP2-F4', 'FP2-F8', 'FPZ', 'FT7', 'FT8',
        'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2',
        'P3', 'P3-O1', 'P4', 'P4-O2', 'P5', 'P6',
        'P7', 'P8', 'PO3', 'PO4', 'PO5', 'PO6',
        'PO7', 'PO8', 'POZ', 'PZ', 'T3', 'T3-C3',
        'T3-T5', 'T4', 'T4-A2', 'T4-T6', 'T5', 'T5-O1',
        'T6', 'T6-O2', 'T7', 'T8', 'TP7', 'TP8',
    )

    def __init__(self, patch_size=40, hid_channels=64, num_queries=4, depth=8,
                 num_heads=2, mlp_ratio=4., drop_path=0.):
        super().__init__()
        for name, value in [('patch_size', patch_size), ('hid_channels', hid_channels),
                            ('num_queries', num_queries), ('depth', depth), ('num_heads', num_heads)]:
            _positive_int(name, value)
        _probability('drop_path', drop_path)
        if (isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)) or
                not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hid_channels * mlp_ratio) < 1):
            raise ValueError('mlp_ratio must give a positive finite hidden width.')
        if patch_size < 8 or patch_size % 4 or hid_channels % 16 or hid_channels % num_heads or (hid_channels // num_heads) % 2:
            raise ValueError('Require patch_size >=8 divisible by4, hid_channels divisible by16, and even head width.')
        self.patch_size = patch_size
        self.hid_channels = hid_channels
        self.num_queries = num_queries
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.feature_dim = hid_channels * num_queries
        self.patch_embed = _PatchEmbedding(hid_channels, patch_size)
        self.freq_embed = _FrequencyEmbedding(hid_channels, patch_size)
        self.channel_location_embedder = nn.Sequential(_Mlp(hid_channels, 2 * hid_channels, hid_channels, norm=True))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hid_channels))
        self.cross_attn = _CrossAttention(hid_channels, num_heads, num_queries, mlp_ratio)
        self.blocks = nn.ModuleList([_RotaryBlock(self.feature_dim, num_heads * num_queries, mlp_ratio, drop_path)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(self.feature_dim)
        self.decoder_head = _ReconstructionHead(hid_channels, num_heads, patch_size, num_queries)
        self.channel_emb = _ChannelEmbedding(hid_channels)
        nn.init.orthogonal_(self.cross_attn.query_embed)
        nn.init.trunc_normal_(self.mask_token, std=.02, a=-.02, b=.02)
        self.apply(_initialize)
        with torch.no_grad():
            for index, block in enumerate(self.blocks):
                block.attn.proj.weight.div_(math.sqrt(2. * (index + 1)))
                block.mlp.fc2.weight.div_(math.sqrt(2. * (index + 1)))

    def _prepare(self, x, positions, mask, position_noise):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                min(x.shape) < 1 or x.shape[-1] % self.patch_size):
            raise ValueError('EEG must be floating [batch, electrodes, samples], nonempty and divisible by patch_size.')
        if not isinstance(positions, torch.Tensor) or not positions.is_floating_point():
            raise ValueError('positions must be a floating tensor in one coordinate frame.')
        if positions.shape == (x.shape[1], 3):
            positions = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        if positions.shape != (x.shape[0], x.shape[1], 3):
            raise ValueError('positions must have shape [electrodes,3] or [batch,electrodes,3].')
        if isinstance(position_noise, bool) or not isinstance(position_noise, (int, float)) or not math.isfinite(position_noise) or position_noise < 0:
            raise ValueError('position_noise must be a finite nonnegative number.')
        batch, channels, samples = x.shape
        patches = samples // self.patch_size
        embedded = self.patch_embed(x) + self.freq_embed(x)
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError('mask must be a boolean patch or sample tensor.')
            if mask.shape == x.shape:
                mask = mask.reshape(batch, channels, patches, self.patch_size).any(-1)
            if mask.shape != (batch, channels, patches):
                raise ValueError('mask must have shape [batch,electrodes,patches] or match EEG samples.')
            embedded = torch.where(mask.to(x.device).flatten(1).unsqueeze(-1), self.mask_token, embedded)
        positions = positions.to(device=x.device, dtype=x.dtype)
        if positions.dtype in (torch.float16, torch.bfloat16):
            positions = positions.float()
        low, high = positions.min(dim=1, keepdim=True)[0], positions.max(dim=1, keepdim=True)[0]
        positions = (positions - low) / (high - low + 1e-8)
        if position_noise:
            positions = positions + torch.randn_like(positions) * position_noise
        position_features = _position_encoding(positions, self.hid_channels).to(x.dtype)
        position_features = self.channel_location_embedder(position_features).repeat_interleave(patches, dim=0)
        tokens = embedded.reshape(batch, channels, patches, self.hid_channels).permute(0, 2, 1, 3).reshape(
            batch * patches, channels, self.hid_channels)
        return tokens + position_features, position_features

    def _encode(self, x, positions, mask, position_noise):
        tokens, position_features = self._prepare(x, positions, mask, position_noise)
        unified, attention = self.cross_attn(tokens)
        latent = unified.reshape(x.shape[0], -1, self.feature_dim)
        for block in self.blocks:
            latent = block(latent)
        return self.norm(latent), position_features, attention

    def forward_features(self, x, positions, mask=None, position_noise=0., return_attention=False):
        r'''Return [batch, patches, feature_dim] and optionally [batch, patches, queries, electrodes] attention.

        position_noise is the standard deviation of optional Gaussian jitter
        after coordinate normalization. Zero (default) is deterministic in eval;
        0.02 selects the native masked-path augmentation. Jitter is explicit
        and applies whenever requested, independently of train/eval mode.
        '''
        if not isinstance(return_attention, bool):
            raise ValueError('return_attention must be a boolean.')
        latent, _, attention = self._encode(x, positions, mask, position_noise)
        if return_attention:
            return latent, attention.reshape(x.shape[0], latent.shape[1], self.num_queries, x.shape[1])
        return latent

    @classmethod
    def get_channel_ids(cls, names):
        r'''Return native decoder IDs for a sequence of channel names; names are case-insensitive.

        This fixed mapping does not infer channel positions or construct a
        montage. New names outside the native vocabulary cannot be decoded
        through an arbitrary existing row. Encoder features need only positions.
        '''
        if not isinstance(names, (tuple, list)) or not names or not all(isinstance(name, str) for name in names):
            raise ValueError('names must be a nonempty list or tuple of channel names.')
        try:
            return [cls.channel_names.index(name.upper()) for name in names]
        except ValueError as error:
            raise ValueError('Unknown LUNA native decoder channel name.') from error

    def _channel_ids(self, channel_ids, batch, channels, device):
        try:
            ids = torch.as_tensor(channel_ids, device=device)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError('channel_ids must contain native integer IDs.') from error
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('channel_ids must contain integers.')
        if ids.shape == (channels,):
            ids = ids.unsqueeze(0).expand(batch, -1)
        if ids.shape != (batch, channels) or torch.any((ids < 0) | (ids >= len(self.channel_names))):
            raise ValueError('channel_ids must be valid native IDs with shape [electrodes] or [batch,electrodes].')
        return ids.long()

    def forward(self, x, positions, channel_ids, mask=None, position_noise=0., return_attention=False):
        latent, position_features, attention = self._encode(x, positions, mask, position_noise)
        ids = self._channel_ids(channel_ids, x.shape[0], x.shape[1], x.device)
        decoder_queries = position_features + self.channel_emb(ids).repeat_interleave(latent.shape[1], dim=0)
        reconstructed = self.decoder_head(latent, decoder_queries)
        if not isinstance(return_attention, bool):
            raise ValueError('return_attention must be a boolean.')
        if return_attention:
            return reconstructed, attention.reshape(x.shape[0], latent.shape[1], self.num_queries, x.shape[1])
        return reconstructed

    @classmethod
    def from_pretrained(cls, checkpoint_path, variant='base', prefix='', drop_path=0.):
        r'''Strictly load the full local official safetensors state, including the reconstruction decoder.

        Variants base/large/huge fix native dimensions and patch_size40, with
        two heads per query. Select the matching official file and checksum.
        Loading is tensor-only, CPU-based and never downloads or executes remote
        code. Optional prefix explicitly selects a complete component. Require
        all keys, exact shapes/dtypes and finite tensors; no weights are skipped.
        Install safetensors or the foundation extra. Source-code Apache-2.0 does
        not change the separately licensed CC-BY-ND-4.0 official weights.
        '''
        if not isinstance(variant, str) or variant not in cls.variants:
            raise ValueError('Unknown LUNA pretrained variant.')
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        _probability('drop_path', drop_path)
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError('LUNA weights require safetensors; install torcheeg[foundation].') from error
        state = load_file(str(Path(checkpoint_path)), device='cpu')
        selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        model = cls(**cls.variants[variant], drop_path=drop_path)
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete LUNA state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(selected, strict=True)
        return model


class _ClassificationHead(nn.Module):
    def __init__(self, width, classes, heads):
        super().__init__()
        self.decoder_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=.15)
        self.decoder_ffn = _Mlp(width, 4 * width, classes, dropout=.15)
        self.learned_agg = nn.Parameter(torch.randn(1, 1, width))
        self.apply(_initialize)

    def forward(self, x):
        queries = self.learned_agg.expand(x.shape[0], -1, -1)
        return self.decoder_ffn(self.decoder_attn(queries, x, x)[0][:, 0])


class LUNAClassifier(nn.Module):
    r'''LUNA native aggregation-query classification head over complete pretrained features.

    Return [batch, num_classes] logits. Official pretraining weights initialize
    the backbone, while the head is newly initialized. The complete native
    reconstruction state is retained for safe restoration but unused by this
    classification forward. Frozen backbone dropout stays disabled in train.

    Args:
        backbone (LUNA): Complete pretrained or randomly initialized backbone.
        num_classes (int): Number of output classes. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
        num_heads (int): Aggregation head attention heads. (default: :obj:`2`)
        positions (torch.Tensor, optional): Fixed [electrodes,3] coordinates,
            stored as a buffer for ordinary model(eeg) trainer calls. Per-call
            coordinates override this buffer. (default: :obj:`None`)
    '''
    def __init__(self, backbone, num_classes=2, freeze_backbone=False, num_heads=2, positions=None):
        super().__init__()
        if not isinstance(backbone, LUNA):
            raise ValueError('backbone must be a LUNA model.')
        _positive_int('num_classes', num_classes)
        _positive_int('num_heads', num_heads)
        if backbone.feature_dim % num_heads or not isinstance(freeze_backbone, bool):
            raise ValueError('num_heads must divide feature_dim and freeze_backbone must be boolean.')
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = _ClassificationHead(backbone.feature_dim, num_classes, num_heads).to(backbone.mask_token)
        if positions is not None:
            if (not isinstance(positions, torch.Tensor) or not positions.is_floating_point() or
                    positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] != 3 or
                    not torch.isfinite(positions).all()):
                raise ValueError('Fixed positions must be finite floating [electrodes,3] coordinates.')
            positions = positions.detach().clone().to(backbone.mask_token)
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
        return self.classifier(self.backbone.forward_features(x, positions))
