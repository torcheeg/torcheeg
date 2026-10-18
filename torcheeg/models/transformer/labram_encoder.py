# Adapted from the MIT-licensed LaBraMEncoder implementation by Wei-Bang Jiang.
# See LICENSES/LaBraM.txt and LICENSES/LaBraM-NOTICE.txt.
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int, _probability
from ._checkpoint import _load_checkpoint


_CHANNEL_NAMES = ('FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', 'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2')


class _DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class _TemporalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, (1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv3 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, channels)
        self.gelu3 = nn.GELU()

    def forward(self, x):
        batch, electrodes, patches, samples = x.shape
        x = x.reshape(batch, 1, electrodes * patches, samples)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return x.permute(0, 2, 3, 1).reshape(batch, electrodes * patches, -1)


class _Mlp(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class _Attention(nn.Module):
    def __init__(self, width, heads, qkv_bias, qk_norm, norm_eps, dropout, attention_dropout):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(width)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(width)) if qkv_bias else None
        self.q_norm = nn.LayerNorm(width // heads, eps=norm_eps) if qk_norm else None
        self.k_norm = nn.LayerNorm(width // heads, eps=norm_eps) if qk_norm else None
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(width, width)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(x)
        if self.q_bias is not None:
            qkv = qkv + torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = qkv.reshape(batch, tokens, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        # A two-dimensional normalization view avoids x86/ARM Inductor
        # backward fusion conflicts without changing native per-head math.
        if self.q_norm is not None:
            q = self.q_norm(q.flatten(0, 2)).reshape_as(q).to(v.dtype)
            k = self.k_norm(k.flatten(0, 2)).reshape_as(k).to(v.dtype)
        attention = ((q * (width // self.num_heads) ** -.5) @ k.transpose(-2, -1)).softmax(dim=-1)
        x = self.attn_drop(attention) @ v
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, width)))


class _Block(nn.Module):
    def __init__(self, width, hidden, heads, qkv_bias, qk_norm, norm_eps, dropout, attention_dropout, drop_path, layer_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.attn = _Attention(width, heads, qkv_bias, qk_norm, norm_eps, dropout, attention_dropout)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        self.mlp = _Mlp(width, hidden, dropout)
        self.gamma_1 = nn.Parameter(torch.full((width,), layer_scale)) if layer_scale > 0 else None
        self.gamma_2 = nn.Parameter(torch.full((width,), layer_scale)) if layer_scale > 0 else None

    def forward(self, x):
        update = self.attn(self.norm1(x))
        x = x + self.drop_path(update if self.gamma_1 is None else self.gamma_1 * update)
        update = self.mlp(self.norm2(x))
        return x + self.drop_path(update if self.gamma_2 is None else self.gamma_2 * update)


class LaBraMEncoder(nn.Module):
    r'''LaBraMEncoder raw-EEG encoder from Jiang et al. (ICLR 2024).

    Supply real floating [batch, electrodes, samples] windows or explicit
    [batch, electrodes, patches, patch_size] patches, plus integer channel
    indices in source order. Native input is 200 Hz, microvolts divided by
    100, with one-second/200-sample patches. Filtering, reference selection,
    scaling and resampling belong to the caller's data pipeline.

    The temporal convolution produces channel-major patch tokens. All
    native encoder tensor names, including the learned final norm, remain
    intact. forward returns final norm AFTER mean pooling, excluding CLS;
    pooling='cls' selects the normalized CLS token instead. forward_all and
    forward_features expose normalized or raw tokens explicitly. Mean of
    normalized tokens differs from normalization of their mean.

    .. code-block:: python

        import torch
        from torcheeg.models import LaBraMEncoder

        model = LaBraMEncoder()
        indices = model.channel_indices(['Fp1', 'Fp2', 'Cz'])
        features = model(torch.randn(2, 3, 800), indices)

    .. automethod:: channel_indices
    .. automethod:: forward_all
    .. automethod:: forward_features
    .. automethod:: from_pretrained

    Args:
        patch_size (int): Native samples per patch. (default: :obj:`200`)
        hid_channels (int): Width; must equal ceil(patch_size/8) * conv_channels. (default: :obj:`200`)
        conv_channels (int): Temporal convolution channels, divisible by four. (default: :obj:`8`)
        mlp_channels (int): Feed-forward width. (default: :obj:`800`)
        depth (int): Transformer blocks. (default: :obj:`12`)
        num_heads (int): Attention heads dividing hid_channels. (default: :obj:`10`)
        qkv_bias (bool): Native asymmetric query/value biases. (default: :obj:`False`)
        qk_norm (bool): Normalize query/key head features. (default: :obj:`True`)
        dropout (float): Token, projection and final MLP dropout. (default: :obj:`0.0`)
        attention_dropout (float): Attention probability dropout. (default: :obj:`0.0`)
        drop_path_rate (float): Maximum stochastic-depth probability. (default: :obj:`0.0`)
        layer_scale (float): Positive residual scales; zero omits them. (default: :obj:`0.1`)
        norm_eps (float): LayerNorm epsilon. (default: :obj:`1e-6`)
        use_abs_pos_emb (bool): Use native 129-slot CLS/channel table. (default: :obj:`True`)
        max_patches (int): Learned temporal table size. (default: :obj:`16`)
        interpolate_time (bool): Explicitly interpolate the learned table for longer windows. This is an extension beyond native <=16-patch inference. (default: :obj:`False`)
        pooling (str): 'mean' or 'cls'. (default: :obj:`'mean'`)
    '''
    def __init__(self, patch_size=200, hid_channels=200, conv_channels=8, mlp_channels=800,
                 depth=12, num_heads=10, qkv_bias=False, qk_norm=True, dropout=0.,
                 attention_dropout=0., drop_path_rate=0., layer_scale=.1, norm_eps=1e-6,
                 use_abs_pos_emb=True, max_patches=16, interpolate_time=False, pooling='mean'):
        super().__init__()
        for name, value in [('patch_size', patch_size), ('hid_channels', hid_channels), ('conv_channels', conv_channels),
                            ('mlp_channels', mlp_channels), ('depth', depth), ('num_heads', num_heads), ('max_patches', max_patches)]:
            _positive_int(name, value)
        if conv_channels % 4 or (patch_size + 7) // 8 * conv_channels != hid_channels or hid_channels % num_heads:
            raise ValueError('Convolution channels must divide into four groups, match the flattened width and support all attention heads.')
        for name, value in [('qkv_bias', qkv_bias), ('qk_norm', qk_norm), ('use_abs_pos_emb', use_abs_pos_emb), ('interpolate_time', interpolate_time)]:
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean.')
        for name, value in [('dropout', dropout), ('attention_dropout', attention_dropout), ('drop_path_rate', drop_path_rate)]:
            _probability(name, value)
        for name, value in [('layer_scale', layer_scale), ('norm_eps', norm_eps)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name == 'norm_eps' and value == 0):
                raise ValueError(f'{name} must be finite and positive (or zero for layer_scale).')
        if pooling not in ('mean', 'cls'):
            raise ValueError('pooling must be mean or cls.')
        self.patch_size = patch_size
        self.hid_channels = hid_channels
        self.max_patches = max_patches
        self.interpolate_time = interpolate_time
        self.pooling = pooling
        self.patch_embed = _TemporalConv(conv_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_channels))
        self.pos_embed = nn.Parameter(torch.zeros(1, 129, hid_channels)) if use_abs_pos_emb else None
        self.time_embed = nn.Parameter(torch.zeros(1, max_patches, hid_channels))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(hid_channels, mlp_channels, num_heads, qkv_bias, qk_norm,
            norm_eps, dropout, attention_dropout, drop_path_rate * i / max(depth - 1, 1), layer_scale) for i in range(depth)])
        self.norm = nn.LayerNorm(hid_channels, eps=norm_eps)
        self.apply(self._initialize)
        for value in (self.cls_token, self.pos_embed, self.time_embed):
            if value is not None:
                nn.init.trunc_normal_(value, std=.02)
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                block.attn.proj.weight.div_(math.sqrt(2 * (i + 1)))
                block.mlp.fc2.weight.div_(math.sqrt(2 * (i + 1)))

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def channel_indices(channel_names):
        r'''Resolve known native names to slots 1..128; CLS slot zero is internal.

        Case and outer whitespace are ignored. Native bipolar names have
        their own distinct slots; no derivation or reference suffix is
        silently converted to a monopolar name. Preserve source order.
        '''
        if not isinstance(channel_names, (list, tuple)) or not channel_names:
            raise ValueError('channel_names must be a nonempty ordered list/tuple.')
        names = [name.strip().upper() if isinstance(name, str) else '' for name in channel_names]
        table = {name.upper(): i + 1 for i, name in enumerate(_CHANNEL_NAMES)}
        if any(name not in table or table[name] > 128 for name in names) or len(set(names)) != len(names):
            raise ValueError('Use unique native channel names within the checkpoint 128 electrode slots; the last eight catalog derivations have no pretrained slot.')
        return torch.tensor([table[name] for name in names], dtype=torch.long)

    def _patches(self, eeg):
        if not isinstance(eeg, torch.Tensor) or not eeg.is_floating_point() or eeg.ndim not in (3, 4):
            raise ValueError('EEG must be real floating [batch, electrodes, samples] or explicit four-dimensional patches.')
        if eeg.shape[0] < 1 or eeg.shape[1] < 1 or eeg.shape[-1] < 1:
            raise ValueError('EEG dimensions must be nonempty.')
        if eeg.ndim == 3:
            if eeg.shape[-1] % self.patch_size:
                raise ValueError('Every window must contain complete patches; partial samples are not discarded.')
            eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], -1, self.patch_size)
        if eeg.shape[-1] != self.patch_size or eeg.shape[2] < 1:
            raise ValueError('Explicit patches must match patch_size and contain at least one patch.')
        if not self.interpolate_time and eeg.shape[2] > self.max_patches:
            raise ValueError('Window exceeds the temporal table; enable interpolate_time explicitly for this extension.')
        return eeg

    def forward_all(self, eeg, channel_indices, normalize=True):
        r'''Return [batch, 1 + electrodes * patches, width] tokens, CLS first.

        Patch tokens use electrode-major, temporal-minor order. normalize=True
        applies the learned native final norm separately to every token.
        '''
        if not isinstance(normalize, bool):
            raise ValueError('normalize must be a boolean.')
        eeg = self._patches(eeg)
        if not isinstance(channel_indices, torch.Tensor) or channel_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('channel_indices must be an int32/int64 tensor.')
        if channel_indices.ndim == 1:
            channel_indices = channel_indices.unsqueeze(0)
        if channel_indices.ndim != 2 or channel_indices.shape[1] != eeg.shape[1] or channel_indices.shape[0] not in (1, eeg.shape[0]):
            raise ValueError('Supply [electrodes], [1, electrodes] or [batch, electrodes] channel slots.')
        indices = channel_indices.to(device=eeg.device).expand(eeg.shape[0], -1)
        indices = torch.where((indices < 1) | (indices > 128), 129, indices)
        patches = eeg.shape[2]
        x = self.patch_embed(eeg)
        temporal = self.time_embed
        if patches > self.max_patches:
            temporal = F.interpolate(temporal.transpose(1, 2), size=patches, mode='linear', align_corners=False).transpose(1, 2)
        temporal = temporal[:, :patches].unsqueeze(1).expand(eeg.shape[0], eeg.shape[1], -1, -1).flatten(1, 2)
        cls = self.cls_token.expand(eeg.shape[0], -1, -1)
        if self.pos_embed is not None:
            spatial = F.embedding(indices, self.pos_embed[0]).unsqueeze(2).expand(-1, -1, patches, -1).flatten(1, 2)
            x = x + spatial
            cls = cls + self.pos_embed[:, :1]
        else:
            # Keep channel-index validation active in the position ablation.
            zero_spatial = F.embedding(indices, x.new_zeros((129, 1))).unsqueeze(2).expand(-1, -1, patches, -1).flatten(1, 2)
            x = x + zero_spatial
        x = x + temporal
        x = self.pos_drop(torch.cat((cls, x), dim=1))
        for block in self.blocks:
            x = block(x)
        return self.norm(x) if normalize else x

    def forward_features(self, eeg, channel_indices, normalize=True):
        r'''Return the channel-major [batch, electrodes, patches, width] grid.'''
        tokens = self.forward_all(eeg, channel_indices, normalize=normalize)[:, 1:]
        patches = eeg.shape[2] if eeg.ndim == 4 else eeg.shape[-1] // self.patch_size
        return tokens.reshape(eeg.shape[0], eeg.shape[1], patches, self.hid_channels)

    def forward(self, eeg, channel_indices):
        tokens = self.forward_all(eeg, channel_indices, normalize=False)
        pooled = tokens[:, 1:].mean(dim=1) if self.pooling == 'mean' else tokens[:, 0]
        return self.norm(pooled)

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', exclude_pretraining=False, allow_training_metadata=False, **kwargs):
        r'''Load a complete local native encoder with explicit settings.

        Accept tensor/state_dict/model containers; require all encoder keys,
        shapes and finite floating tensors. prefix='student.' selects the
        native pretraining student. exclude_pretraining=True discards only
        that student's mask_token and lm_head tensors. No norm/fc_norm rename,
        silent partial load, download or unsafe pickle fallback occurs.

        The verified author training file contains Namespace and NumPy scalar
        metadata. Explicit allow_training_metadata scopes only those known
        types through safe_globals, always with weights_only=True. This
        requires the public safe_globals context (tested 2.6/2.14); convert
        a verified file to tensor-only state before using PyTorch 2.4.
        '''
        if not isinstance(prefix, str) or not isinstance(exclude_pretraining, bool) or not isinstance(allow_training_metadata, bool):
            raise ValueError('Use a string prefix and boolean loading options.')
        checkpoint = _load_checkpoint(Path(checkpoint_path), allow_training_metadata)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must be a tensor dictionary or supported container.')
        containers = [name for name in ('model', 'state_dict') if name in checkpoint]
        if len(containers) > 1:
            raise ValueError('Ambiguous model/state_dict checkpoint containers.')
        state = checkpoint[containers[0]] if containers else checkpoint
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('State keys must be strings.')
        selected = {key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
        if exclude_pretraining:
            selected = {key:value for key,value in selected.items() if key != 'mask_token' and not key.startswith('lm_head.')}
        model = cls(**kwargs)
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete LaBraMEncoder encoder state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name,value in selected.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != expected[name].shape:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(selected, strict=True)
        return model
