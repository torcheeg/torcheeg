import math
import warnings
from collections.abc import Mapping
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ._checkpoint import _load_checkpoint


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()

    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (
                2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid(
                [coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - \
                1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(
                    size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Preserve per-head math while avoiding backward fusion layout conflicts.
        if self.q_norm is not None:
            q = self.q_norm(q.flatten(0, 2)).reshape_as(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k.flatten(0, 2)).reshape_as(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(
                x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x),
                                   rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 *
                                   self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(
            1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans,
                               kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans,
                               kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class LaBraM(nn.Module):
    '''
    Implementation of Large Brain Model (LaBraM) for EEG signal processing.
    
    - Paper: Jiang W, Zhao L, Lu B. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI[C]//The Twelfth International Conference on Learning Representations.
    - URL: https://openreview.net/forum?id=QzTpTRVtrP
    - Related Project: https://github.com/935963004/LaBraM/

    Below is a quick start example:

    .. code-block:: python

        model = LaBraM.base_patch200_200(num_classes=4)
        # batch_size, num_electrodes, chunk_size // patch_size, patch_size
        x = torch.randn(2, 6, 8, 200)
        model(x, electrodes=['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5'])

    .. automethod:: load_pretrained

    Args:
        chunk_size (int): The total length of the EEG signal segment to process. (default: :obj:`1600`)
        patch_size (int): The size of each temporal patch. (default: :obj:`200`)
        out_chans (int): Number of output channels from the temporal convolution. (default: :obj:`8`)
        num_classes (int): Number of classes for classification. (default: :obj:`1000`)
        embed_dim (int): Dimension of the embedding space. (default: :obj:`200`)
        depth (int): Number of transformer layers. (default: :obj:`12`)
        num_heads (int): Number of attention heads in each transformer layer. (default: :obj:`10`)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. (default: :obj:`4.0`)
        qkv_bias (bool): If True, add a learnable bias to query, key, value. (default: :obj:`False`)
        qk_norm (callable): Normalization layer for query and key. (default: :obj:`nn.LayerNorm`)
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. (default: :obj:`None`)
        drop_rate (float): Dropout rate. (default: :obj:`0.0`)
        attn_drop_rate (float): Attention dropout rate. (default: :obj:`0.0`)
        drop_path_rate (float): Stochastic depth rate. (default: :obj:`0.0`)
        norm_layer (callable): Normalization layer. (default: :obj:`nn.LayerNorm`)
        init_values (float): Initial values for layer scale. (default: :obj:`0.0`)
        use_mean_pooling (bool): If True, use mean pooling for final feature vector. (default: :obj:`True`)
        init_scale (float): Initial scale for the head layer. (default: :obj:`0.001`)
        use_abs_pos_emb (bool): If True, use absolute positional embeddings. (default: :obj:`True`)
        **kwargs: Additional keyword arguments
    '''
    def __init__(self, chunk_size=1600, patch_size=200, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0,
                 use_mean_pooling=True, init_scale=0.001, use_abs_pos_emb=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.time_window = chunk_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(
            1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, electrodes=[], return_patch_tokens=False, return_all_tokens=False, **kwargs):
        input_chans = None
        if len(electrodes):
            for electrode in electrodes:
                assert electrode in standard_1020, f"{electrode} not in standard_1020"
            assert len(electrodes) == x.shape[1], f"Number of electrodes {len(electrodes)} should match input ({x.shape[1]})."
            input_chans = get_input_chans(electrodes)
        else:
            assert len(standard_1020) == x.shape[1], f"You must provide electrodes for the input. Expected default channels {standard_1020} are used."

        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if input_chans is None or max(input_chans) >= self.pos_embed.shape[1]:
                raise ValueError('Provide explicit electrodes within the learned spatial table; the full catalog exceeds its 128 electrode slots.')
            pos_embed_used = self.pos_embed[:, input_chans]
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
                batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(
                batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(
                1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, electrodes=[], return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(
            x, electrodes=electrodes, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def load_pretrained(self, checkpoint_path, *, strict=False, prefix=None,
                        exclude_head=False, allow_training_metadata=False):
        r'''Restore local legacy-classifier weights with explicit loading reports.

        Accept a tensor state dictionary or one model/module/state_dict
        container. By default select student. when present; otherwise retain
        raw state keys. An explicit prefix selects and strips exactly that
        prefix. Empty or wholly unrelated selections fail before mutation.

        strict=False permits transfer loading and warns about every missing
        or unexpected key. strict=True requires complete classifier state.
        Matching tensors must have valid shapes, real floating values and
        finite entries before any parameter is copied. exclude_head=True
        deliberately leaves the classification head fresh. Native norm and
        downstream fc_norm are not renamed: their mismatch is reported.

        Deserialization always uses weights_only=True. Explicit metadata
        opt-in scopes only the known author Namespace/NumPy scalar types;
        runtimes without safe_globals require verified tensor-only conversion.
        No download or unsafe pickle fallback occurs. This method belongs to
        the existing classification API, separate from LaBraMEncoder.

        Args:
            checkpoint_path (str or file-like): Local checkpoint accepted by torch.load.
            strict (bool): Require complete matching classifier state. (default: :obj:`False`)
            prefix (str, optional): Explicit prefix; None automatically selects student. or raw keys. (default: :obj:`None`)
            exclude_head (bool): Remove head.* tensors before transfer loading. (default: :obj:`False`)
            allow_training_metadata (bool): Explicit known-author metadata opt-in. (default: :obj:`False`)

        Returns:
            NamedTuple: The missing_keys and unexpected_keys returned by load_state_dict.
        '''
        if not isinstance(strict, bool) or not isinstance(exclude_head, bool):
            raise ValueError('strict and exclude_head must be booleans.')
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError('prefix must be a string or None.')
        checkpoint = _load_checkpoint(checkpoint_path, allow_training_metadata)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must be a tensor dictionary or supported container.')
        containers = [key for key in ('model', 'module', 'state_dict') if key in checkpoint]
        if len(containers) > 1:
            raise ValueError('Ambiguous checkpoint containers; supply one state dictionary.')
        state = checkpoint[containers[0]] if containers else checkpoint
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        if prefix is None:
            prefix = 'student.' if any(key.startswith('student.') for key in state) else ''
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if exclude_head:
            selected = {key: value for key, value in selected.items() if not key.startswith('head.')}
        if not selected or not all(isinstance(value, torch.Tensor) for value in selected.values()):
            raise ValueError('Selected checkpoint state must be a nonempty tensor dictionary.')
        expected = self.state_dict()
        matching = set(expected) & set(selected)
        if not matching:
            raise ValueError('Selected checkpoint has no matching classifier tensors.')
        missing, extra = sorted(set(expected) - set(selected)), sorted(set(selected) - set(expected))
        for key in matching:
            value = selected[key]
            if not value.is_floating_point() or value.shape != expected[key].shape:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {key}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {key}.')
        if missing or extra:
            message = f'Incomplete LaBraM classifier state: missing={missing}, unexpected={extra}.'
            if strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        return self.load_state_dict(selected, strict=strict)

    @staticmethod
    def base_patch200_200(**kwargs):
        return LaBraM(
            patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1, **kwargs)

    @staticmethod
    def large_patch200_200(**kwargs):
        return LaBraM(
            patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=1e-5, **kwargs)

    @staticmethod
    def huge_patch200_200(**kwargs):
        return LaBraM(
            patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=1e-5, **kwargs)
