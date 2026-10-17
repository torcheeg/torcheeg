# Adapted from BINE022/EEGPT under Apache-2.0; see LICENSES/EEGPT.txt.
import math

import torch
from torch import nn
from torch.nn import functional as F

from .eegpt import _Attention, _DropPath, _MLP, _positive_int, _probability


def _rotate(x, frequencies):
    width = frequencies.shape[-1]
    if width == 0:
        return x
    left, right = x[..., :width], x[..., width:]
    pairs = left.reshape(*left.shape[:-1], width // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    frequencies = frequencies.to(dtype=x.dtype)
    return torch.cat((left * frequencies.cos() + rotated * frequencies.sin(), right), dim=-1)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim, interpolate_factor):
        super().__init__()
        frequencies = 1. / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim))
        self.freqs = nn.Parameter(frequencies, requires_grad=False)
        self.interpolate_factor = interpolate_factor

    def forward(self, patches, summaries, x):
        positions = torch.arange(patches, device=x.device, dtype=self.freqs.dtype)
        positions = positions.repeat_interleave(summaries) / self.interpolate_factor
        positions = torch.cat((positions.new_zeros(1), positions))
        return torch.outer(positions, self.freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)


class _RotaryAttention(_Attention):
    def forward(self, x, frequencies, return_attention=False):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.num_heads,
                                     width // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _rotate(q, frequencies), _rotate(k, frequencies)
        if return_attention:
            return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1)
        x = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.attn_dropout if self.training else 0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, width)))


class _RotaryBlock(nn.Module):
    def __init__(self, width, heads, mlp_width, qkv_bias, dropout, attn_dropout, drop_path, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.attn = _RotaryAttention(width, heads, qkv_bias, dropout, attn_dropout)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        self.mlp = _MLP(width, mlp_width, dropout)

    def forward(self, x, frequencies, return_attention=False):
        attention = self.attn(self.norm1(x), frequencies, return_attention)
        if return_attention:
            return attention
        x = x + self.drop_path(attention)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _Decoder(nn.Module):
    def __init__(self, kind, num_patches, in_features, embed_num, hid_channels,
                 output_features, depth, num_heads, mlp_ratio, qkv_bias,
                 dropout, attn_dropout, drop_path, norm_eps, input_projection,
                 position_embedding, interpolate_factor, init_std, return_attention_layer):
        super().__init__()
        for name, value in [('num_patches', num_patches), ('in_features', in_features),
                            ('embed_num', embed_num), ('hid_channels', hid_channels),
                            ('output_features', output_features), ('depth', depth), ('num_heads', num_heads)]:
            _positive_int(name, value)
        for name, value in [('dropout', dropout), ('attn_dropout', attn_dropout), ('drop_path', drop_path)]:
            _probability(name, value)
        for name, value in [('mlp_ratio', mlp_ratio), ('norm_eps', norm_eps), ('init_std', init_std),
                            ('interpolate_factor', interpolate_factor)]:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if interpolate_factor < 1 or hid_channels % num_heads or int(hid_channels * mlp_ratio) < 1:
            raise ValueError('Require interpolate_factor >= 1, divisible attention width and a nonempty MLP.')
        for value in (qkv_bias, input_projection, position_embedding):
            if not isinstance(value, bool):
                raise ValueError('qkv_bias, input_projection and position_embedding must be booleans.')
        if not input_projection and in_features != hid_channels:
            raise ValueError('Without input_projection, in_features must equal hid_channels.')
        if (isinstance(return_attention_layer, bool) or not isinstance(return_attention_layer, int) or
                not (return_attention_layer == -1 or 1 <= return_attention_layer <= depth)):
            raise ValueError('return_attention_layer must be -1 or a one-based block index.')
        self.num_patches = num_patches
        self.in_features = in_features
        self.embed_num = embed_num
        self.hid_channels = hid_channels
        self.output_features = output_features
        self.depth = depth
        self.num_heads = num_heads
        self.use_inp_embed = input_projection
        self.use_pos_embed = position_embedding
        self.return_attention_layer = return_attention_layer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_channels))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        if input_projection:
            setattr(self, kind + '_embed', nn.Linear(in_features, hid_channels))
        if position_embedding:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_num, hid_channels))
            nn.init.trunc_normal_(self.pos_embed, std=init_std)
        mask_shape = (1, 1, hid_channels) if kind == 'reconstructor' else (1, 1, embed_num, hid_channels)
        self.mask_token = nn.Parameter(torch.zeros(mask_shape))
        self.time_embed = _TimeEmbedding((hid_channels // num_heads) // 2, interpolate_factor)
        if kind == 'reconstructor':
            # Retained for native checkpoint compatibility; not used by this
            # downstream full-token forward (neither is mask_token).
            self.chan_embed = nn.Embedding(62, hid_channels)
        rates = torch.linspace(0, drop_path, depth).tolist()
        blocks = nn.ModuleList([_RotaryBlock(hid_channels, num_heads, int(hid_channels * mlp_ratio),
            qkv_bias, dropout, attn_dropout, rates[index], norm_eps) for index in range(depth)])
        setattr(self, kind + '_blocks', blocks)
        setattr(self, kind + '_norm', nn.LayerNorm(hid_channels, eps=norm_eps))
        setattr(self, kind + '_proj', nn.Linear(hid_channels, output_features))
        if kind == 'predictor':
            nn.init.trunc_normal_(self.mask_token, std=init_std)

    def _check_input(self, x, use_out_proj):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1:] != (self.num_patches, self.embed_num, self.in_features):
            raise ValueError('Input must match [batch, num_patches, embed_num, in_features].')
        if not isinstance(use_out_proj, bool):
            raise ValueError('use_out_proj must be a boolean.')

    def _prepare_tokens(self, x):
        if self.use_pos_embed:
            x = x + self.pos_embed
        frequencies = self.time_embed(self.num_patches, self.embed_num, x)
        tokens = x.flatten(1, 2)
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        return tokens, frequencies.to(dtype=tokens.dtype)


class EEGPTReconstructor(_Decoder):
    r'''EEGPT temporal reconstructor for full encoder summary sequences.

    Input is [batch, num_patches, embed_num, in_features]. A class token is
    prepended to the flattened patch/summary axis. Temporal attention applies
    the official partial-head rotary position encoding, with all summaries in
    a patch sharing its temporal position. Output is [batch, 1 + patches *
    summaries, hid_channels], or patch_size features with use_out_proj=True.

    This is the official downstream full-token reconstructor, not a masked
    pretraining loss or a waveform reconstruction pipeline. mask_token and
    chan_embed tensors are retained for native checkpoint compatibility but
    are unused in this forward. Projection norm/weights are used only when
    requested. No dummy forward or mutable frequency cache is created.

    - Reference: https://github.com/BINE022/EEGPT
    - License: Apache-2.0, retained in LICENSES/EEGPT.txt.

    Args:
        num_patches (int): Number of temporal encoder patches. (default: :obj:`16`)
        patch_size (int): Projected features per output token. (default: :obj:`64`)
        in_features (int): Width of each encoder summary. (default: :obj:`512`)
        embed_num (int): Summary tokens per temporal patch. (default: :obj:`4`)
        hid_channels (int): Temporal transformer width. (default: :obj:`512`)
        depth (int): Temporal transformer blocks. (default: :obj:`8`)
        num_heads (int): Attention heads. (default: :obj:`8`)
        mlp_ratio (float): MLP expansion ratio. (default: :obj:`4.0`)
        qkv_bias (bool): Attention Q/K/V bias. (default: :obj:`True`)
        dropout (float): Projection/MLP dropout. (default: :obj:`0.0`)
        attn_dropout (float): Attention dropout. (default: :obj:`0.0`)
        drop_path (float): Maximum stochastic depth rate. (default: :obj:`0.0`)
        norm_eps (float): LayerNorm epsilon. (default: :obj:`1e-6`)
        input_projection (bool): Map encoder width into temporal width. (default: :obj:`True`)
        position_embedding (bool): Add learned summary-slot embeddings. (default: :obj:`False`)
        interpolate_factor (float): Rotary time-position divisor, at least one. (default: :obj:`2.0`)
        init_std (float): Optional position-embedding initialization scale. (default: :obj:`0.02`)
        return_attention_layer (int): One-based attention inspection index, or -1. (default: :obj:`-1`)
    '''
    def __init__(self, num_patches=16, patch_size=64, in_features=512, embed_num=4,
                 hid_channels=512, depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 dropout=0., attn_dropout=0., drop_path=0., norm_eps=1e-6,
                 input_projection=True, position_embedding=False, interpolate_factor=2.,
                 init_std=.02, return_attention_layer=-1):
        super().__init__('reconstructor', num_patches, in_features, embed_num, hid_channels,
            patch_size, depth, num_heads, mlp_ratio, qkv_bias, dropout, attn_dropout,
            drop_path, norm_eps, input_projection, position_embedding, interpolate_factor,
            init_std, return_attention_layer)

    def forward(self, x, use_out_proj=False):
        self._check_input(x, use_out_proj)
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)
        x, frequencies = self._prepare_tokens(x)
        for index, block in enumerate(self.reconstructor_blocks, 1):
            inspect = index == self.return_attention_layer
            x = block(x, frequencies, return_attention=inspect)
            if inspect:
                return x
        return self.reconstructor_proj(self.reconstructor_norm(x)) if use_out_proj else x


class EEGPTPredictor(_Decoder):
    r'''EEGPT temporal predictor with optional projection to encoder width.

    The input/output token layout and temporal rotary positions follow
    EEGPTReconstructor. With use_out_proj=True, the predictor returns
    in_features per token; otherwise it returns hid_channels. Its mask_token
    is retained for checkpoint compatibility but is unused in this downstream
    full-token forward. This module supplies no masked pretraining objective.

    - Reference: https://github.com/BINE022/EEGPT
    - License: Apache-2.0, retained in LICENSES/EEGPT.txt.

    Args:
        num_patches (int): Number of temporal encoder patches. (default: :obj:`16`)
        in_features (int): Width of each encoder summary and projected output. (default: :obj:`512`)
        embed_num (int): Summary tokens per temporal patch. (default: :obj:`4`)
        hid_channels (int): Temporal transformer width. (default: :obj:`512`)
        depth (int): Temporal transformer blocks. (default: :obj:`8`)
        num_heads (int): Attention heads. (default: :obj:`8`)
        mlp_ratio (float): MLP expansion ratio. (default: :obj:`4.0`)
        qkv_bias (bool): Attention Q/K/V bias. (default: :obj:`True`)
        dropout (float): Projection/MLP dropout. (default: :obj:`0.0`)
        attn_dropout (float): Attention dropout. (default: :obj:`0.0`)
        drop_path (float): Maximum stochastic depth rate. (default: :obj:`0.0`)
        norm_eps (float): LayerNorm epsilon. (default: :obj:`1e-6`)
        input_projection (bool): Map encoder width into temporal width. (default: :obj:`True`)
        position_embedding (bool): Add learned summary-slot embeddings. (default: :obj:`False`)
        interpolate_factor (float): Rotary time-position divisor, at least one. (default: :obj:`2.0`)
        init_std (float): Mask/optional position-embedding initialization scale. (default: :obj:`0.02`)
        return_attention_layer (int): One-based attention inspection index, or -1. (default: :obj:`-1`)
    '''
    def __init__(self, num_patches=16, in_features=512, embed_num=4, hid_channels=512,
                 depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True, dropout=0.,
                 attn_dropout=0., drop_path=0., norm_eps=1e-6, input_projection=True,
                 position_embedding=False, interpolate_factor=2., init_std=.02,
                 return_attention_layer=-1):
        super().__init__('predictor', num_patches, in_features, embed_num, hid_channels,
            in_features, depth, num_heads, mlp_ratio, qkv_bias, dropout, attn_dropout,
            drop_path, norm_eps, input_projection, position_embedding, interpolate_factor,
            init_std, return_attention_layer)

    def forward(self, x, use_out_proj=False):
        self._check_input(x, use_out_proj)
        if self.use_inp_embed:
            x = self.predictor_embed(x)
        x, frequencies = self._prepare_tokens(x)
        for index, block in enumerate(self.predictor_blocks, 1):
            inspect = index == self.return_attention_layer
            x = block(x, frequencies, return_attention=inspect)
            if inspect:
                return x
        return self.predictor_proj(self.predictor_norm(x)) if use_out_proj else x
