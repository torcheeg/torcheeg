# ST-EEGFormer adaptation; see LICENSES/STEEGFormer.txt and its notice.
# Copyright (c) 2025 LIUYIN YANG.
import math
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability


_CHANNEL_INDICES = {
    'C1': 0, 'PZ': 1, 'C4': 2, 'F6': 3, 'FTT8H': 4,
    'OZ': 5, 'FP1': 6, 'FCC5H': 7, 'TPP8H': 8, 'CPP6H': 9,
    'C2': 10, 'F4': 11, 'OI2H': 12, 'AF4': 13, 'FCZ': 14,
    'CCP6H': 15, 'TP8': 16, 'POO10H': 17, 'FC1': 18, 'FC6': 19,
    'C5': 20, 'P8': 21, 'FT8': 22, 'P6': 23, 'P9': 24,
    'FZ': 25, 'AFF1': 26, 'TPP10H': 27, 'AFF2': 28, 'P10': 29,
    'CPP2H': 30, 'M1': 31, 'FCC6H': 32, 'FTT7H': 33, 'FC2': 34,
    'PPO2': 35, 'AFP3H': 36, 'AF7': 37, 'PO10': 38, 'AF8': 39,
    'CPP1H': 40, 'P7': 41, 'F1': 42, 'AFP4H': 43, 'PO9': 44,
    'FT9': 45, 'CP2': 46, 'IZ': 47, 'FCC1H': 48, 'FC5': 49,
    'T5': 50, 'CP5': 51, 'CP6': 52, 'FFC5H': 53, 'F2': 54,
    'M2': 55, 'POO9H': 56, 'AFF5H': 57, 'PO4': 58, 'POO3H': 59,
    'FP2': 60, 'T3': 61, 'CP4': 62, 'POZ': 63, 'TTP7H': 64,
    'T7': 65, 'A2': 66, 'CCP4H': 67, 'T8': 68, 'PPO10H': 69,
    'FC3': 70, 'F3': 71, 'F5': 72, 'A1': 73, 'P3': 74,
    'FC4': 75, 'FCC2H': 76, 'FFC6H': 77, 'FFT8H': 78, 'CCP2H': 79,
    'CPP4H': 80, 'T6': 81, 'FTT9H': 82, 'PPO6H': 83, 'CP3': 84,
    'CP1': 85, 'AF3': 86, 'FT10': 87, 'OI1H': 88, 'TPP9H': 89,
    'P5': 90, 'I2': 91, 'CCP1H': 92, 'T4': 93, 'CCP3H': 94,
    'O1': 95, 'PO5': 96, 'PPO9H': 97, 'PPO5H': 98, 'P1': 99,
    'AFZ': 100, 'PO6': 101, 'PO3': 102, 'O2': 103, 'CPP5H': 104,
    'FFC1H': 105, 'FCC4H': 106, 'FFT7H': 107, 'FFC2H': 108, 'FFC4H': 109,
    'CZ': 110, 'TP7': 111, 'FPZ': 112, 'FTT10H': 113, 'PO7': 114,
    'CPP3H': 115, 'P4': 116, 'P2': 117, 'F8': 118, 'CPZ': 119,
    'FCC3H': 120, 'FFC3H': 121, 'FT7': 122, 'I1': 123, 'TTP8H': 124,
    'AFF6H': 125, 'CCP5H': 126, 'C6': 127, 'PPO1': 128, 'PO8': 129,
    'C3': 130, 'POO4H': 131, 'TPP7H': 132, 'F7': 133, 'T9': 134,
    'TP9': 135, 'T10': 136, 'TP10': 137, 'POO1': 138, 'POO2': 139,
    'PPO1H': 140, 'PPO2H': 141,
}


class _PatchEmbedding(nn.Module):
    def __init__(self, patch_size, width):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, width)

    def forward(self, eeg):
        return self.proj(eeg.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3))


class _ChannelEmbedding(nn.Module):
    def __init__(self, count, width):
        super().__init__()
        self.channel_transformation = nn.Embedding(count, width)
        nn.init.zeros_(self.channel_transformation.weight)

    def forward(self, indices):
        return self.channel_transformation(indices)


class _TemporalEmbedding(nn.Module):
    def __init__(self, width, max_patches):
        super().__init__()
        position = torch.arange(max_patches + 1).unsqueeze(1).float()
        frequency = torch.exp(torch.arange(0, width, 2).float() * -(math.log(10000.) / width))
        pe = torch.zeros(1, max_patches + 1, width)
        pe[0, :, 0::2] = torch.sin(position * frequency)
        pe[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer('pe', pe)


class _DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0.:
            return x
        keep = 1. - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(keep + torch.rand(shape, dtype=x.dtype, device=x.device))
        return x / keep * mask


class _Attention(nn.Module):
    def __init__(self, width, heads, qkv_bias, dropout, attention_dropout):
        super().__init__()
        self.num_heads = heads
        self.attention_dropout = attention_dropout
        self.qkv = nn.Linear(width, 3 * width, bias=qkv_bias)
        self.proj = nn.Linear(width, width)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.num_heads,
            width // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.)
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(batch, tokens, width)))


class _MLP(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, width)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class _Block(nn.Module):
    def __init__(self, width, heads, hidden, qkv_bias, dropout, attention_dropout, drop_path, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.attn = _Attention(width, heads, qkv_bias, dropout, attention_dropout)
        self.drop_path1 = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        self.mlp = _MLP(width, hidden, dropout)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class STEEGFormer(nn.Module):
    r'''ST-EEGFormer encoder with explicit native feature-normalization modes.

    - Paper: Yang L, Sun Q, Li A, et al. Are EEG Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse BCI Tasks. ICLR 2026.
    - URL: https://openreview.net/forum?id=5Xwm8e6vbh
    - Related Project: https://github.com/LiuyinYang1101/STEEGFormer

    Supply real floating EEG [batch, electrodes, samples] at the intended
    sampling rate and explicit learned channel indices. Native pretraining
    uses 128 Hz, 16-sample patches and recommends windows up to six seconds.
    This encoder performs no filtering, resampling or signal normalization.
    It rejects incomplete patches rather than silently dropping samples.

    forward returns the mean of normalized patch tokens, retaining the
    archived encoder interface. forward_all includes CLS. forward_features
    returns [batch, patches, electrodes, hid_channels]. Use normalize=False
    explicitly for the author's downstream mean-pooling protocol, which
    omits the final encoder norm. The two representations are different.
    The pretraining MAE decoder is not included.

    .. code-block:: python

        import torch
        from torcheeg.models import STEEGFormer

        model = STEEGFormer()
        indices = model.channel_indices(['Fp1', 'Fp2', 'Cz'])
        output = model(torch.randn(2, 3, 128), indices)

    Args:
        patch_size (int): Samples per temporal patch. (default: :obj:`16`)
        hid_channels (int): Encoder feature width. (default: :obj:`512`)
        mlp_channels (int): Feed-forward hidden width. (default: :obj:`2048`)
        depth (int): Number of transformer blocks. (default: :obj:`8`)
        num_heads (int): Attention heads dividing hid_channels. (default: :obj:`8`)
        num_channel_embeddings (int): Learned channel table size. The canonical table has 145 slots and 142 named channels. (default: :obj:`145`)
        max_patches (int): Maximum patch positions, excluding CLS position zero. (default: :obj:`511`)
        qkv_bias (bool): Include packed query/key/value bias. (default: :obj:`True`)
        dropout (float): Embedding, attention-output and MLP dropout. (default: :obj:`0.0`)
        attention_dropout (float): Attention probability dropout. (default: :obj:`0.0`)
        drop_path_rate (float): Maximum linearly scheduled stochastic depth. (default: :obj:`0.0`)
        norm_eps (float): Layer normalization epsilon. (default: :obj:`1e-6`)
    '''
    def __init__(self, patch_size=16, hid_channels=512, mlp_channels=2048, depth=8,
                 num_heads=8, num_channel_embeddings=145, max_patches=511,
                 qkv_bias=True, dropout=0., attention_dropout=0., drop_path_rate=0., norm_eps=1e-6):
        super().__init__()
        for name, value in [('patch_size', patch_size), ('hid_channels', hid_channels),
                            ('mlp_channels', mlp_channels), ('depth', depth), ('num_heads', num_heads),
                            ('num_channel_embeddings', num_channel_embeddings), ('max_patches', max_patches)]:
            _positive_int(name, value)
        if hid_channels % 2 or hid_channels % num_heads:
            raise ValueError('hid_channels must be even and divisible by num_heads.')
        if not isinstance(qkv_bias, bool):
            raise ValueError('qkv_bias must be a boolean.')
        for name, value in [('dropout', dropout), ('attention_dropout', attention_dropout), ('drop_path_rate', drop_path_rate)]:
            _probability(name, value)
        if (isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float))
                or not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError('norm_eps must be finite and positive.')
        self.patch_size = patch_size
        self.hid_channels = hid_channels
        self.depth = depth
        self.num_channel_embeddings = num_channel_embeddings
        self.max_patches = max_patches
        self.patch_embed = _PatchEmbedding(patch_size, hid_channels)
        self.enc_channel_emd = _ChannelEmbedding(num_channel_embeddings, hid_channels)
        self.enc_temporal_emd = _TemporalEmbedding(hid_channels, max_patches)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hid_channels))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(hid_channels, num_heads, mlp_channels, qkv_bias,
            dropout, attention_dropout, drop_path_rate * index / max(depth - 1, 1), norm_eps)
            for index in range(depth)])
        self.norm = nn.LayerNorm(hid_channels, eps=norm_eps)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def channel_indices(names):
        r'''Resolve exact pretrained electrode names, ignoring letter case.

        Return a CPU long tensor in the supplied order. Unknown names,
        reference suffixes and bipolar derivations require explicit user
        preprocessing; they are not silently reassigned to another sensor.
        The remaining unnamed table slots require explicit integer indices.
        '''
        if (not isinstance(names, (tuple, list)) or not names or
                any(not isinstance(name, str) or name.strip().upper() not in _CHANNEL_INDICES for name in names)):
            raise ValueError('names must be a nonempty list/tuple of known pretrained channel names.')
        return torch.tensor([_CHANNEL_INDICES[name.strip().upper()] for name in names], dtype=torch.long)

    def forward_all(self, eeg, channel_indices, normalize=True):
        r'''Return all tokens [batch, 1 + patches * electrodes, hid_channels].

        Tokens use temporal-major, channel-minor order after the leading CLS.
        normalize=True applies the native pretraining encoder's final norm.
        False returns the pre-norm features used in official downstream pooling.
        '''
        if (not isinstance(eeg, torch.Tensor) or not eeg.is_floating_point() or eeg.ndim != 3
                or min(eeg.shape) < 1 or eeg.shape[-1] % self.patch_size
                or not 1 <= eeg.shape[-1] // self.patch_size <= self.max_patches):
            raise ValueError('EEG must be real floating [batch, electrodes, samples] with complete patches within max_patches.')
        if not isinstance(normalize, bool):
            raise ValueError('normalize must be a boolean.')
        if (not isinstance(channel_indices, torch.Tensor) or channel_indices.dtype not in (torch.int32, torch.int64)
                or channel_indices.ndim not in (1, 2) or channel_indices.shape[-1] != eeg.shape[1]):
            raise ValueError('channel_indices must be an integer tensor [electrodes], [1, electrodes] or [batch, electrodes].')
        if channel_indices.ndim == 1:
            channel_indices = channel_indices.unsqueeze(0)
        if channel_indices.shape[0] not in (1, eeg.shape[0]):
            raise ValueError('Channel-index batch size must be one or match the EEG batch.')
        channel_indices = channel_indices.to(device=eeg.device).expand(eeg.shape[0], -1)
        # Inductor can wrap negative embedding indices. Map every invalid
        # index to the positive out-of-bounds sentinel in both execution modes.
        channel_indices = torch.where((channel_indices < 0) | (channel_indices >= self.num_channel_embeddings),
                                      self.num_channel_embeddings, channel_indices)
        channels = self.enc_channel_emd(channel_indices).unsqueeze(1)
        x = self.patch_embed(eeg)
        batch, patches, electrodes, width = x.shape
        temporal = self.enc_temporal_emd.pe[:, 1:patches + 1].unsqueeze(2)
        x = (x + temporal + channels).reshape(batch, patches * electrodes, width)
        cls = (self.cls_token + self.enc_temporal_emd.pe[:, :1]).expand(batch, -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1))
        for block in self.blocks:
            x = block(x)
        return self.norm(x) if normalize else x

    def forward_features(self, eeg, channel_indices, normalize=True):
        r'''Return patch tokens [batch, patches, electrodes, hid_channels].'''
        tokens = self.forward_all(eeg, channel_indices, normalize=normalize)[:, 1:]
        return tokens.reshape(eeg.shape[0], eeg.shape[-1] // self.patch_size, eeg.shape[1], self.hid_channels)

    def forward(self, eeg, channel_indices):
        return self.forward_all(eeg, channel_indices)[:, 1:].mean(dim=1)

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', exclude_decoder=False, allow_namespace=False, **kwargs):
        r'''Safely load a complete local encoder with explicit constructor settings.

        Accept a tensor dictionary, state_dict checkpoint or native model
        checkpoint. Reject ambiguous containers. Require all encoder keys,
        exact shapes, floating and finite tensors. Other prefix components
        are ignored. exclude_decoder=True explicitly discards recognized MAE
        decoder components; this does not restore a complete MAE.

        Native training files can contain argparse.Namespace metadata.
        allow_namespace=True permits only that known standard-library class
        in a local safe_globals context. Loading always uses weights_only=True,
        never an unsafe fallback, and does not change the global allowlist.
        This opt-in requires the public safe_globals context API (available
        in tested PyTorch 2.6/2.14). On 2.4, first convert the verified file
        to a tensor-only encoder state on a supported runtime.
        No data, weights or preprocessing settings are downloaded or inferred.
        '''
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        if not isinstance(exclude_decoder, bool) or not isinstance(allow_namespace, bool):
            raise ValueError('exclude_decoder and allow_namespace must be booleans.')
        context = nullcontext()
        if allow_namespace:
            if not hasattr(torch.serialization, 'safe_globals'):
                raise RuntimeError('This runtime lacks safe_globals; convert the verified checkpoint to tensor-only encoder state first.')
            if Namespace not in torch.serialization.get_safe_globals():
                context = torch.serialization.safe_globals([Namespace])
        with context:
            checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        containers = [key for key in ('state_dict', 'model') if key in checkpoint]
        if len(containers) > 1:
            raise ValueError('Checkpoint has ambiguous model/state_dict containers.')
        state = checkpoint[containers[0]] if containers else checkpoint
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if exclude_decoder:
            decoder = ('decoder_embed.', 'decoder_blocks.', 'decoder_norm.', 'decoder_pred.',
                       'dec_channel_emd.', 'dec_temporal_emd.')
            selected = {key: value for key, value in selected.items() if key != 'mask_token' and not key.startswith(decoder)}
        model = cls(**kwargs)
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete ST-EEGFormer encoder state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name, value in selected.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != expected[name].shape:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(selected, strict=True)
        return model


