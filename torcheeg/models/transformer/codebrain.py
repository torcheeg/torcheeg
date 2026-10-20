"""CodeBrain EEGSSM, adapted from jingyingma01/CodeBrain (Apache-2.0).

Authors: Jingying Ma et al. Source revision22d350caf68246d2fda4f630ef837420db3fb130.
See LICENSES/CodeBrain.txt. Global convolution follows SGConv (Li et al.,2022).
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ._utils import _positive_int, _probability


class _Kernel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(8, width, 2))


class _TransposedLinear(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 4 * width))
        self.bias = nn.Parameter(torch.empty(width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(4 * width), 1 / math.sqrt(4 * width))

    def forward(self, x):
        # Preserve the author's contraction order without opt_einsum at runtime.
        return torch.tensordot(self.weight, x, dims=([1], [1])).permute(1, 0, 2) + self.bias[:, None]


class _GlobalConvolution(nn.Module):
    def __init__(self, width, max_length, dropout):
        super().__init__()
        self.D = nn.Parameter(torch.randn(4, width))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout else nn.Identity()
        self.norm = nn.Identity()
        self.output_linear = _TransposedLinear(width)
        self.num_scales = 1 + math.ceil(math.log2(max_length / 2))
        self.kernel_list = nn.ModuleList([_Kernel(width) for _ in range(self.num_scales)])
        self.register_buffer('multiplier', torch.full((1, width, 1), 2.))
        self.register_buffer('kernel_norm', torch.ones(8, width, 1))
        self.register_buffer('kernel_norm_initialized', torch.tensor(False))
        self._kernel_norm_ready = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._kernel_norm_ready = bool(self.kernel_norm_initialized.item())

    def forward(self, x):
        length = x.shape[-1]
        pieces = [F.interpolate(block.kernel, scale_factor=2 ** max(0, index - 1), mode='linear') *
                  self.multiplier ** (self.num_scales - index - 1)
                  for index, block in enumerate(self.kernel_list)]
        kernel = torch.cat(pieces, dim=-1)
        # Like the official state, cache the initial full-kernel norm once.
        # It depends on weights, not EEG or batch statistics, and is saved.
        if not self._kernel_norm_ready:
            self.kernel_norm.copy_(kernel.norm(dim=-1, keepdim=True).detach())
            self.kernel_norm_initialized.fill_(True)
            self._kernel_norm_ready = True
        kernel = kernel[..., :length] if kernel.shape[-1] > length else F.pad(kernel, (0, length-kernel.shape[-1]))
        kernel = kernel / self.kernel_norm
        left, right = kernel.reshape(2, 4, kernel.shape[1], length).unbind(0)
        kernel = F.pad(left, (0, length)) + F.pad(right.flip(-1), (length, 0))
        spectrum = torch.fft.rfft(kernel.float(), n=2 * length)
        signal = torch.fft.rfft(x.float(), n=2 * length)
        # opt_einsum's native contraction multiplies the kernel first. Complex
        # multiplication order can affect float32 rounding, so retain it.
        output = torch.fft.irfft(spectrum.unsqueeze(0) * signal.unsqueeze(1), n=2 * length)[..., :length]
        output = output + self.D[None, :, :, None] * x.unsqueeze(1)
        output = output.flatten(1, 2)
        return self.output_linear(self.norm(self.dropout(self.activation(output))))


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, width, 1))

    def forward(self, x):
        return self.scale * (x / (x.norm(dim=1, keepdim=True) / math.sqrt(x.shape[1]) + 1e-8))


def _weight_norm_conv(input_width, output_width, kernel_size):
    return nn.utils.weight_norm(nn.Conv1d(input_width, output_width, kernel_size, padding=(kernel_size-1)//2))


class _Conv(nn.Module):
    def __init__(self, input_width, output_width, kernel_size=3):
        super().__init__()
        self.conv = _weight_norm_conv(input_width, output_width, kernel_size)

    def forward(self, x):
        return self.conv(x)


class _FinalConv(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.conv = nn.Conv1d(input_width, output_width, 1)

    def forward(self, x):
        return self.conv(x)


class _ResidualBlock(nn.Module):
    def __init__(self, width, max_length, dropout):
        super().__init__()
        self.width = width
        self.sn = _RMSNorm(width)
        self.S41 = _GlobalConvolution(2 * width, max_length, dropout)
        self.conv_layer = _Conv(width, 2 * width)
        self.attention = nn.MultiheadAttention(2 * width, 4, dropout=dropout, batch_first=True)
        self.gelu = nn.GELU()
        self.res_conv = _weight_norm_conv(width, width, 1)
        self.skip_conv = _weight_norm_conv(width, width, 1)

    def forward(self, x, original, return_residual=True):
        normalized = self.sn(x) if return_residual else None
        temporal = self.S41(self.gelu(self.conv_layer(x + original)))
        spatial = temporal.transpose(1, 2)
        # The author uses a window of one: only the diagonal is visible.
        mask = torch.full((x.shape[-1], x.shape[-1]), float('-inf'), device=x.device, dtype=spatial.dtype)
        mask.fill_diagonal_(0)
        spatial = self.attention(spatial, spatial, spatial, attn_mask=mask)[0].transpose(1, 2)
        gated = temporal + spatial
        gated = gated[:, :self.width].tanh() * gated[:, self.width:].sigmoid()
        residual = (normalized + self.res_conv(gated)) * math.sqrt(.5) if return_residual else None
        return residual, self.skip_conv(gated)


class _ResidualGroup(nn.Module):
    def __init__(self, width, depth, max_length, dropout):
        super().__init__()
        self.residual_blocks = nn.ModuleList([_ResidualBlock(width, max_length, dropout) for _ in range(depth)])

    def forward(self, x):
        original = x
        skip = 0
        for index, block in enumerate(self.residual_blocks):
            # The final residual is discarded by the author. Do not attach its
            # unused parameters to a compiled graph; retain their saved state.
            x, part = block(x, original, return_residual=index < len(self.residual_blocks) - 1)
            skip = skip + part
        return skip * math.sqrt(1. / len(self.residual_blocks))


class _PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoding = nn.Sequential(nn.Conv2d(200, 200, (19, 7), padding=(9, 3), groups=200))
        self.mask_encoding = nn.Parameter(torch.zeros(200), requires_grad=False)
        self.proj_in = nn.Sequential(nn.Conv2d(1, 25, (1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(25, 25, (1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(25, 25, (1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, 200), nn.Dropout(.1))

    def forward(self, x, mask):
        batch, channels, patches, size = x.shape
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_encoding, x)
        embedded = self.proj_in(x.reshape(batch, 1, channels * patches, size))
        embedded = embedded.permute(0, 2, 1, 3).reshape(batch, channels, patches, 200)
        spectrum = torch.fft.rfft(x.reshape(batch * channels * patches, size), norm='forward').abs()
        embedded = embedded + self.spectral_proj(spectrum.reshape(batch, channels, patches, 101))
        return embedded + self.positional_encoding(embedded.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class CodeBrain(nn.Module):
    r'''CodeBrain EEGSSM pretrained encoder with time/frequency token prediction.

    Input [batch,electrodes,patches,200] contains nonoverlapping one-second
    patches at 200 Hz. Output preserves all four dimensions, including batch1
    and single electrodes/patches. No scaling, filtering, montage alignment or
    normalization is inferred. Channel order matters to positional convolution.
    Both pretrained4096-way prediction heads are retained. This class is the
    EEGSSM encoder; the separate TFDual tokenizer is not included.

    Args:
        hid_channels (int): Residual and skip width. (default: :obj:`200`)
        depth (int): Residual blocks. (default: :obj:`8`)
        dropout (float): Global-convolution/attention dropout. (default: :obj:`0.1`)
        max_length (int): Determines the full kernel scale count. (default: :obj:`570`)
        num_time_tokens (int): Time-domain vocabulary size. (default: :obj:`4096`)
        num_frequency_tokens (int): Frequency-domain vocabulary size. (default: :obj:`4096`)
    '''
    def __init__(self, hid_channels=200, depth=8, dropout=.1, max_length=570,
                 num_time_tokens=4096, num_frequency_tokens=4096):
        super().__init__()
        for name, value in [('hid_channels', hid_channels), ('depth', depth), ('max_length', max_length),
                            ('num_time_tokens', num_time_tokens), ('num_frequency_tokens', num_frequency_tokens)]:
            _positive_int(name, value)
        if hid_channels % 2 or max_length < 2:
            raise ValueError('hid_channels must be even and max_length at least2.')
        _probability('dropout', dropout)
        self.hid_channels = hid_channels
        self.depth = depth
        self.dropout = dropout
        self.max_length = max_length
        self.num_time_tokens = num_time_tokens
        self.num_frequency_tokens = num_frequency_tokens
        self.feature_dim = 200
        self.patch_embedding = _PatchEmbedding()
        self.init_conv = nn.Sequential(_Conv(200, hid_channels, 1), nn.ReLU())
        self.residual_layer = _ResidualGroup(hid_channels, depth, max_length, dropout)
        self.final_conv = nn.Sequential(_Conv(hid_channels, hid_channels, 1), nn.ReLU(), _FinalConv(hid_channels, 200))
        self.lm_head_t = nn.Linear(200, num_time_tokens, bias=False)
        self.lm_head_f = nn.Linear(200, num_frequency_tokens, bias=False)
        self.norm = nn.LayerNorm(200)

    def forward(self, x, mask=None):
        r'''Return [batch,electrodes,patches,200] continuous encoder features.

        Optional boolean [batch,electrodes,patches] mask replaces selected raw
        patches with the frozen native zero token BEFORE convolution/FFT.
        On the first forward, native kernel-norm buffers initialize from the
        full learned kernels and are retained in subsequent state exports.
        '''
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 4 or
                min(x.shape) < 1 or x.shape[-1] != 200):
            raise ValueError('EEG must be nonempty floating [batch,electrodes,patches,200].')
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != x.shape[:-1]:
                raise ValueError('mask must be boolean [batch,electrodes,patches].')
            mask = mask.to(x.device)
        embedded = self.patch_embedding(x, mask)
        sequence = embedded.permute(0, 3, 1, 2).reshape(x.shape[0], 200, -1)
        output = self.final_conv(self.residual_layer(self.init_conv(sequence)))
        output = output.transpose(1, 2).reshape_as(x)
        return self.norm(output)

    def predict_tokens(self, x, mask):
        r'''Return time/frequency logits [selected_patches,vocabulary] for masked patches.'''
        if mask is None:
            raise ValueError('Token prediction requires a boolean mask.')
        features = self(x, mask)
        selected = features[mask.to(x.device)]
        return self.lm_head_t(selected), self.lm_head_f(selected)

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='module.'):
        r'''Strictly load every tensor from the local official EEGSSM checkpoint.

        The official file is a flat tensor state with explicit module. prefix.
        Use prefix='' for an unprefixed complete export. Tensor-only CPU loading
        uses weights_only=True, with no unsafe fallback, network or skipped
        tensors. Both vocabulary heads and initial kernel buffers are required.
        '''
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        state = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Require a flat tensor state dictionary.')
        selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        model = cls()
        expected = model.state_dict()
        if set(selected) != set(expected):
            raise ValueError(f'Incomplete CodeBrain state: missing={sorted(set(expected)-set(selected))}, '
                             f'extra={sorted(set(selected)-set(expected))}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
            if name.endswith('kernel_norm') and (value <= 0).any():
                raise ValueError(f'Kernel norm must be positive: {name}.')
        model.load_state_dict(selected, strict=True)
        return model


class CodeBrainClassifier(nn.Module):
    r'''CodeBrain downstream flatten/ELU classification head with complete backbone.

    Args:
        backbone (CodeBrain): Complete EEGSSM backbone.
        num_electrodes (int): Expected electrode count. (default: :obj:`19`)
        num_patches (int): Expected patches per input. (default: :obj:`4`)
        num_classes (int): Output logits for cross-entropy. (default: :obj:`2`)
        dropout (float): Head dropout. (default: :obj:`0.1`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_electrodes=19, num_patches=4, num_classes=2,
                 dropout=.1, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, CodeBrain) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a CodeBrain backbone and boolean freeze_backbone.')
        for name, value in [('num_electrodes', num_electrodes), ('num_patches', num_patches), ('num_classes', num_classes)]:
            _positive_int(name, value)
        _probability('dropout', dropout)
        self.backbone = backbone
        self.num_electrodes = num_electrodes
        self.num_patches = num_patches
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Sequential(nn.Linear(num_electrodes * num_patches * 200, max(4, num_patches) * backbone.hid_channels),
            nn.ELU(), nn.Dropout(dropout), nn.Linear(max(4, num_patches) * backbone.hid_channels, backbone.hid_channels),
            nn.ELU(), nn.Dropout(dropout), nn.Linear(backbone.hid_channels, num_classes)).to(backbone.norm.weight)
        if freeze_backbone:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1:3] != (self.num_electrodes, self.num_patches):
            raise ValueError('EEG electrode/patch dimensions must match the classifier configuration.')
        return self.classifier(self.backbone(x).flatten(1))
