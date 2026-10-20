"""EEGMamba, adapted from Jiquan Wang (MIT), source dbc83fa072744201e8897aeb9f65007b952ad323.

Mamba2/SSD follows Tri Dao and Albert Gu (Apache-2.0), source
8ffd905c91d207f5c0cc84fc2a2fb748655094f0. Modified to use differentiable
PyTorch operations without CUDA extensions. See LICENSES/EEGMamba.txt and Mamba.txt.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ._utils import _positive_int, _probability


def _segment_sum(x):
    length = x.shape[-1]
    values = x.unsqueeze(-1).expand(*x.shape, length)
    lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril(-1)
    values = values.masked_fill(~lower, 0).cumsum(-2)
    diagonal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    return values.masked_fill(~diagonal, -torch.inf)


def _ssd_scan(x, dt, a, b, c, block_size=64):
    # Exact SSD recurrence with bounded intra-block matrices and linear state carry.
    batch, length, heads, head_dim = x.shape
    padding = (-length) % block_size
    x = F.pad(x * dt.unsqueeze(-1), (0, 0, 0, 0, 0, padding))
    decay = F.pad(dt * a, (0, 0, 0, padding))
    b, c = [F.pad(value, (0, 0, 0, padding)) for value in (b, c)]
    blocks = x.shape[1] // block_size
    x = x.reshape(batch, blocks, block_size, heads, head_dim)
    decay = decay.reshape(batch, blocks, block_size, heads).permute(0, 3, 1, 2)
    b, c = [value.reshape(batch, blocks, block_size, 1, -1).expand(-1, -1, -1, heads, -1) for value in (b, c)]
    cumulative = decay.cumsum(-1)
    local_decay = _segment_sum(decay).exp().permute(0, 2, 1, 3, 4)
    coefficients = torch.einsum('bclhn,bcshn->bchls', c, b) * local_decay
    diagonal = torch.einsum('bchls,bcshp->bclhp', coefficients, x)
    end_decay = (cumulative[..., -1:] - cumulative).exp().permute(0, 2, 3, 1)
    local_states = torch.einsum('bclhn,bclh,bclhp->bchpn', b, end_decay, x)
    state = torch.zeros_like(local_states[:, 0])
    outputs = []
    for index in range(blocks):
        off_diagonal = torch.einsum('blhn,bhpn->blhp', c[:, index], state)
        off_diagonal = off_diagonal * cumulative[:, :, index].exp().transpose(1, 2).unsqueeze(-1)
        outputs.append(diagonal[:, index] + off_diagonal)
        state = state * cumulative[:, :, index, -1].exp()[:, :, None, None] + local_states[:, index]
    return torch.stack(outputs, dim=1).reshape(batch, -1, heads, head_dim)[:, :length]


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = 1e-5

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(dtype)


class _Mamba2(nn.Module):
    def __init__(self, width, state_size, head_dim, block_size):
        super().__init__()
        self.inner = 2 * width
        self.state_size = state_size
        self.head_dim = head_dim
        self.heads = self.inner // head_dim
        self.block_size = block_size
        self.in_proj = nn.Linear(width, 2 * self.inner + 2 * state_size + self.heads, bias=False)
        conv_width = self.inner + 2 * state_size
        self.conv1d = nn.Conv1d(conv_width, conv_width, 4, padding=3, groups=conv_width)
        dt = torch.exp(torch.rand(self.heads) * math.log(100) + math.log(.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        for parameter in [self.dt_bias, self.A_log, self.D]:
            parameter._no_weight_decay = True
        self.norm = _RMSNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        z, xbc, dt = self.in_proj(x).split([self.inner, self.inner + 2 * self.state_size, self.heads], dim=-1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = xbc.split([self.inner, self.state_size, self.state_size], dim=-1)
        x = x.reshape(batch, length, self.heads, self.head_dim)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        y = _ssd_scan(x.float(), dt, -self.A_log.float().exp(), b.float(), c.float(), self.block_size)
        y = y + x.float() * self.D.float()[None, None, :, None]
        # The official gated RMSNorm applies SiLU(z) before normalization.
        y = y.flatten(2) * F.silu(z.float())
        return self.out_proj(self.norm(y).to(z.dtype))


class _MixerBlock(nn.Module):
    def __init__(self, width, state_size, head_dim, block_size):
        super().__init__()
        self.norm = _RMSNorm(width)
        self.mixer = _Mamba2(width, state_size, head_dim, block_size)

    def forward(self, x, residual):
        residual = x if residual is None else x + residual
        return self.mixer(self.norm(residual)), residual.float()


class _MixerModel(nn.Module):
    def __init__(self, depth, width, state_size, head_dim, block_size):
        super().__init__()
        self.layers = nn.ModuleList([_MixerBlock(width, state_size, head_dim, block_size) for _ in range(depth)])
        self.norm_f = _RMSNorm(width)

    def forward(self, x):
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
            # EEGMamba reverses BOTH branches after each layer, not just once.
            x, residual = x.flip(1), residual.flip(1)
        return self.norm_f(x + residual)


class _PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_encoding = nn.Parameter(torch.zeros(200), requires_grad=False)
        self.proj_in = nn.Sequential(nn.Conv2d(1, 25, (1, 49), stride=(1, 25), padding=(0, 24), bias=False),
                                     nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, 200, bias=False), nn.Dropout(.1))
        self.positional_encoding = nn.Sequential(nn.Conv2d(200, 200, 7, padding=3, groups=200, bias=False))

    def forward(self, x, mask=None):
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_encoding.to(x.dtype), x)
        batch, channels, patches, samples = x.shape
        temporal = self.proj_in(x.reshape(batch, 1, channels * patches, samples))
        temporal = temporal.permute(0, 2, 1, 3).reshape(batch, channels, patches, 200)
        spectral = self.spectral_proj(torch.fft.rfft(x, dim=-1, norm='forward').abs())
        tokens = temporal + spectral
        return tokens + self.positional_encoding(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class EEGMamba(nn.Module):
    r'''Complete EEGMamba encoder and masked-waveform reconstruction head.

    Adapted from Wang et al., An EEG foundation model with Mamba (2025).
    Input is ``[batch,electrodes,patches,200]`` at 200 Hz. No preprocessing
    is inferred. Electrode-major patches are flattened, and both residual
    and mixer sequences reverse after each of the twelve Mamba2 layers.
    The portable PyTorch SSD path supports gradients without CUDA extensions.

    Args:
        num_layers (int): Mamba2 block count. Official weights use 12. (default: :obj:`12`)
        state_size (int): SSM state size. Official weights use 64. (default: :obj:`64`)
        head_dim (int): Inner-width head size, a divisor of 400. (default: :obj:`50`)
        block_size (int): SSD computation block size, without weight changes. (default: :obj:`64`)
    '''
    def __init__(self, num_layers=12, state_size=64, head_dim=50, block_size=64):
        super().__init__()
        for name, value in [('num_layers', num_layers), ('state_size', state_size),
                            ('head_dim', head_dim), ('block_size', block_size)]:
            _positive_int(name, value)
        if 400 % head_dim:
            raise ValueError('head_dim must divide the inner width 400.')
        self.patch_embedding = _PatchEmbedding()
        self.encoder = _MixerModel(num_layers, 200, state_size, head_dim, block_size)
        self.proj_out = nn.Sequential(nn.Linear(200, 200))
        self.num_layers, self.state_size, self.head_dim, self.block_size = num_layers, state_size, head_dim, block_size
        self.apply(_initialize)

    def _validate(self, x, mask):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.is_floating_point():
            raise ValueError('EEG must be a floating tensor [batch,electrodes,patches,200].')
        if any(size < 1 for size in x.shape[:3]) or x.shape[-1] != 200:
            raise ValueError('EEG needs nonempty batch/electrode/patch axes and exactly 200 samples per patch.')
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
                                 mask.shape != x.shape[:-1] or mask.device != x.device):
            raise ValueError('mask must be boolean [batch,electrodes,patches] on the EEG device.')

    def forward_features(self, x, mask=None):
        r'''Return contextual patch features ``[batch,electrodes,patches,200]``.'''
        self._validate(x, mask)
        batch, channels, patches, _ = x.shape
        tokens = self.patch_embedding(x, mask)
        return self.encoder(tokens.reshape(batch, channels * patches, 200)).reshape(batch, channels, patches, 200)

    def forward(self, x, mask=None):
        r'''Return waveform reconstruction ``[batch,electrodes,patches,200]``.'''
        return self.proj_out(self.forward_features(x, mask))

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', block_size=64):
        r'''Strictly load every tensor from a local official or complete export.

        Uses weights_only=True on CPU. No network, remote execution, unsafe
        pickle fallback or partial state matching. The reconstruction head
        and frozen mask encoding are required; classifiers are separate.
        '''
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        path = Path(checkpoint_path)
        if path.suffix == '.safetensors':
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise ImportError('Install torcheeg[foundation] to load safetensors.') from error
            state = load_file(str(path), device='cpu')
        else:
            state = torch.load(path, weights_only=True, map_location='cpu')
        if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Require a complete flat tensor state dictionary.')
        if prefix and any(not name.startswith(prefix) for name in state):
            raise ValueError('Every state key must start with the explicit prefix.')
        selected = {name[len(prefix):]: value for name, value in state.items()}
        model = cls(block_size=block_size)
        expected = model.state_dict()
        if set(selected) != set(expected):
            raise ValueError(f'Incomplete EEGMamba state: missing={sorted(set(expected)-set(selected))}, '
                             f'extra={sorted(set(selected)-set(expected))}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor shape/type/value mismatch: {name}.')
        model.load_state_dict(selected, strict=True)
        return model


def _initialize(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


class EEGMambaClassifier(nn.Module):
    r'''Downstream flatten/ELU head with the complete EEGMamba backbone.

    Args:
        backbone (EEGMamba): Complete encoder and reconstruction head.
        num_electrodes (int): Expected electrode count. (default: :obj:`19`)
        num_patches (int): Expected temporal patches. (default: :obj:`4`)
        num_classes (int): Number of output logits. (default: :obj:`2`)
        dropout (float): Classifier dropout. (default: :obj:`0.1`)
        freeze_backbone (bool): Train only the classifier. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_electrodes=19, num_patches=4, num_classes=2, dropout=.1, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, EEGMamba) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require an EEGMamba backbone and boolean freeze_backbone.')
        for name, value in [('num_electrodes', num_electrodes), ('num_patches', num_patches), ('num_classes', num_classes)]:
            _positive_int(name, value)
        _probability('dropout', dropout)
        self.backbone = backbone
        self.num_electrodes, self.num_patches, self.freeze_backbone = num_electrodes, num_patches, freeze_backbone
        self.classifier = nn.Sequential(nn.Linear(num_electrodes * num_patches * 200, max(4, num_patches) * 200),
                                        nn.ELU(), nn.Dropout(dropout), nn.Linear(max(4, num_patches) * 200, 200),
                                        nn.ELU(), nn.Dropout(dropout), nn.Linear(200, num_classes)).to(backbone.proj_out[0].weight)
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
        return self.classifier(self.backbone.forward_features(x).flatten(1))
