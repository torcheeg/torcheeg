# Adapted from BINE022/EEGPT, licensed under Apache-2.0.
# See LICENSES/EEGPT.txt for the upstream license and provenance.
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_CHANNEL_NAMES = (
    'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2',
)
_CHANNEL_IDS = {name: index for index, name in enumerate(_CHANNEL_NAMES)}


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')


def _probability(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f'{name} must be in [0, 1).')


def _indices(value, name, ndim, upper, device):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.numel() == 0:
        raise ValueError(f'{name} must be a nonempty {ndim}D integer tensor.')
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'{name} must use int32 or int64 indices.')
    if torch.any(value < 0) or torch.any(value >= upper):
        raise ValueError(f'{name} indices must be in [0, {upper}).')
    return value.to(device=device, dtype=torch.long)


class _DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, device=x.device, dtype=x.dtype)).floor()
        return x / keep * mask


class _MLP(nn.Module):
    def __init__(self, hid_channels, mlp_channels, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_channels, mlp_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_channels, hid_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, hid_channels, num_heads, qkv_bias, dropout, attn_dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(hid_channels, hid_channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(hid_channels, hid_channels)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if return_attention:
            return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(self, hid_channels, num_heads, mlp_channels, qkv_bias,
                 dropout, attn_dropout, drop_path, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(hid_channels, eps=norm_eps)
        self.attn = _Attention(hid_channels, num_heads, qkv_bias, dropout, attn_dropout)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(hid_channels, eps=norm_eps)
        self.mlp = _MLP(hid_channels, mlp_channels, dropout)

    def forward(self, x, return_attention=False):
        attention = self.attn(self.norm1(x), return_attention)
        if return_attention:
            return attention
        x = x + self.drop_path(attention)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, hid_channels, patch_norm):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_norm = patch_norm
        if patch_norm:
            self.proj = nn.Linear(patch_size, hid_channels)
        else:
            self.proj = nn.Conv2d(1, hid_channels, (1, patch_size), stride=(1, patch_stride))

    def forward(self, x):
        if self.patch_norm:
            x = x.unfold(-1, self.patch_size, self.patch_stride).transpose(1, 2)
            return self.proj(F.layer_norm(x, (self.patch_size,)))
        return self.proj(x.unsqueeze(1)).transpose(1, 3)


class EEGPT(nn.Module):
    r'''EEGPT spatial encoder with summary tokens for each temporal patch.

    - Paper: Wang G, Liu W, He Y, et al. EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. NeurIPS, 2024.
    - URL: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
    - Related Project: https://github.com/BINE022/EEGPT

    Input has shape :obj:`[batch, num_electrodes, chunk_size]`. Output has
    shape :obj:`[batch, num_patches, embed_num, hid_channels]`. This encoder
    performs spatial attention independently in each temporal patch. It does
    not contain the pretraining predictor, reconstructor, or a task head.
    Weights are randomly initialized unless :meth:`from_pretrained` is used.
    Explicit electrode names or forward-time channel IDs are required.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT

        model = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        features = model(torch.randn(2, 3, 1024))
        # features.shape == (2, 16, 4, 512)

    Args:
        num_electrodes (int): Number of input electrodes. (default: :obj:`58`)
        chunk_size (int): Number of samples per input window. The official large checkpoint uses four seconds at 256 Hz. (default: :obj:`1024`)
        patch_size (int): Samples in each temporal patch. (default: :obj:`64`)
        patch_stride (int, optional): Temporal stride, or patch_size when omitted. (default: :obj:`None`)
        hid_channels (int): Encoder width. (default: :obj:`512`)
        embed_num (int): Summary tokens per temporal patch. (default: :obj:`4`)
        depth (int): Number of spatial transformer blocks. (default: :obj:`8`)
        num_heads (int): Attention heads per block. (default: :obj:`8`)
        mlp_ratio (float): Hidden MLP width divided by encoder width. (default: :obj:`4.0`)
        qkv_bias (bool): Include attention query/key/value biases. (default: :obj:`True`)
        dropout (float): Projection and MLP dropout probability. (default: :obj:`0.0`)
        attn_dropout (float): Attention dropout probability. (default: :obj:`0.0`)
        drop_path (float): Maximum stochastic depth probability. (default: :obj:`0.0`)
        norm_eps (float): Layer normalization epsilon. (default: :obj:`1e-6`)
        init_std (float): Truncated normal initialization standard deviation. (default: :obj:`0.02`)
        channel_names (list of str, optional): Ordered names in the upstream 62-electrode vocabulary. No implicit resampling, channel mixing or aliases are applied. (default: :obj:`None`)
        patch_norm (bool): Normalize each patch before a linear projection instead of using convolutional patch projection. Changes checkpoint keys/shapes. (default: :obj:`False`)
        return_attention_layer (int): One-based block index whose attention weights are returned; -1 returns summary tokens. (default: :obj:`-1`)
    '''
    def __init__(self, num_electrodes=58, chunk_size=1024, patch_size=64,
                 patch_stride=None, hid_channels=512, embed_num=4, depth=8,
                 num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.0,
                 attn_dropout=0.0, drop_path=0.0, norm_eps=1e-6, init_std=0.02,
                 channel_names=None, patch_norm=False, return_attention_layer=-1):
        super().__init__()
        for name, value in [('num_electrodes', num_electrodes), ('chunk_size', chunk_size),
                            ('patch_size', patch_size), ('hid_channels', hid_channels),
                            ('embed_num', embed_num), ('depth', depth), ('num_heads', num_heads)]:
            _positive_int(name, value)
        patch_stride = patch_size if patch_stride is None else patch_stride
        _positive_int('patch_stride', patch_stride)
        if num_electrodes > len(_CHANNEL_NAMES) or chunk_size < patch_size or hid_channels % num_heads:
            raise ValueError('Require at most 62 electrodes, chunk_size >= patch_size and hid_channels divisible by num_heads.')
        for name, value in [('dropout', dropout), ('attn_dropout', attn_dropout), ('drop_path', drop_path)]:
            _probability(name, value)
        for name, value in [('mlp_ratio', mlp_ratio), ('norm_eps', norm_eps), ('init_std', init_std)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if int(hid_channels * mlp_ratio) < 1:
            raise ValueError('mlp_ratio results in an empty hidden layer.')
        if not isinstance(qkv_bias, bool) or not isinstance(patch_norm, bool):
            raise ValueError('qkv_bias and patch_norm must be booleans.')
        if isinstance(return_attention_layer, bool) or not isinstance(return_attention_layer, int) or return_attention_layer not in [-1, *range(1, depth + 1)]:
            raise ValueError('return_attention_layer must be -1 or a one-based block index.')
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hid_channels = hid_channels
        self.embed_num = embed_num
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (chunk_size - patch_size) // patch_stride + 1
        self.return_attention_layer = return_attention_layer
        self.init_std = init_std
        ids = self.prepare_chan_ids(channel_names) if channel_names is not None else torch.empty(0, dtype=torch.long)
        if ids.numel() and ids.numel() != num_electrodes:
            raise ValueError('channel_names must match num_electrodes.')
        self.register_buffer('channel_ids', ids, persistent=False)
        self.patch_embed = _PatchEmbed(patch_size, patch_stride, hid_channels, patch_norm)
        self.chan_embed = nn.Embedding(len(_CHANNEL_NAMES), hid_channels)
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            _Block(hid_channels, num_heads, int(hid_channels * mlp_ratio), qkv_bias,
                   dropout, attn_dropout, rates[index], norm_eps) for index in range(depth)])
        self.norm = nn.LayerNorm(hid_channels, eps=norm_eps)
        self.summary_token = nn.Parameter(torch.empty(1, embed_num, hid_channels))
        nn.init.trunc_normal_(self.summary_token, std=init_std)
        self.apply(self._init_weights)
        with torch.no_grad():
            for index, block in enumerate(self.blocks, 1):
                block.attn.proj.weight.div_(math.sqrt(2 * index))
                block.mlp.fc2.weight.div_(math.sqrt(2 * index))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def prepare_chan_ids(channels):
        '''Map ordered electrode names to the upstream vocabulary as [1, electrodes].'''
        if isinstance(channels, str) or not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError('channels must be a nonempty list or tuple of electrode names.')
        names = [name.strip().strip('.').upper() if isinstance(name, str) else '' for name in channels]
        if any(name not in _CHANNEL_IDS for name in names) or len(set(names)) != len(names):
            raise ValueError('Electrode names must be unique members of the EEGPT vocabulary.')
        return torch.tensor([[_CHANNEL_IDS[name] for name in names]], dtype=torch.long)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        '''Encode EEG windows or return a selected block's attention weights.

        Args:
            x (torch.Tensor): EEG with shape [batch, electrodes, samples].
            chan_ids (torch.Tensor, optional): Upstream vocabulary IDs with
                shape [electrodes] or [1, electrodes]. Overrides channel_names.
            mask_x (torch.Tensor, optional): Flattened patch/electrode indices
                with shape [retained_patches, retained_electrodes]. A flattened
                index equals patch_index * num_electrodes + electrode_index.
            mask_t (torch.Tensor, optional): Temporal indices into the retained
                patch axis after mask_x, with shape [selected_patches].

        Attention output has shape [batch * retained_patches, heads, tokens,
        tokens], where tokens includes retained electrodes and summary tokens.
        Temporal selection is defined only for summary output.
        '''
        if self.return_attention_layer != -1 and mask_t is not None:
            raise ValueError('mask_t is only supported for summary output.')
        if x.ndim != 3 or x.shape[1:] != (self.num_electrodes, self.chunk_size):
            raise ValueError('EEG input must have shape [batch, num_electrodes, chunk_size].')
        if chan_ids is None:
            chan_ids = self.channel_ids
        if not isinstance(chan_ids, torch.Tensor) or chan_ids.shape not in [(self.num_electrodes,), (1, self.num_electrodes)]:
            raise ValueError('Provide channel_names or chan_ids with one ID per electrode.')
        chan_ids = _indices(chan_ids.reshape(-1), 'chan_ids', 1, len(_CHANNEL_NAMES), x.device)
        if chan_ids.unique().numel() != self.num_electrodes:
            raise ValueError('chan_ids must identify unique electrodes.')
        x = self.patch_embed(x)
        batch, patches, electrodes, width = x.shape
        x = x + self.chan_embed(chan_ids).reshape(1, 1, electrodes, width)
        if mask_x is not None:
            mask_x = _indices(mask_x, 'mask_x', 2, patches * electrodes, x.device)
            x = x.reshape(batch, patches * electrodes, width)[:, mask_x.reshape(-1)]
            patches, electrodes = mask_x.shape
            x = x.reshape(batch, patches, electrodes, width)
        x = x.reshape(batch * patches, electrodes, width)
        summary = self.summary_token.expand(batch * patches, -1, -1)
        x = torch.cat((x, summary), dim=1)
        for index, block in enumerate(self.blocks, 1):
            attention = index == self.return_attention_layer
            x = block(x, return_attention=attention)
            if attention:
                return x
        x = self.norm(x[:, -self.embed_num:]).reshape(batch, patches, self.embed_num, width)
        if mask_t is not None:
            mask_t = _indices(mask_t, 'mask_t', 1, patches, x.device)
            x = x[:, mask_t]
        return x

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='target_encoder.', **kwargs):
        '''Load a complete local encoder tensor state with safe deserialization.

        Accept a direct state dictionary (prefix='') or a Lightning checkpoint's
        state_dict with the explicit prefix. Missing, extra or mismatched encoder
        tensors are errors. Optimizer objects are not allowlisted or executed.
        Architecture and ordered channel names must be supplied by the caller.
        '''
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model = cls(**kwargs)
        expected = model.state_dict()
        missing, extra = set(expected) - set(state), set(state) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete encoder state: missing={sorted(missing)}, extra={sorted(extra)}.')
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != expected[name].shape:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        model.load_state_dict(state, strict=True)
        return model
