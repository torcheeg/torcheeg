# Adapted from Jiaheng Wang's IFNet, revision 00ca53e2 (MIT).
# Copyright (c) 2020 jiaheng-wang. See LICENSES/IFNet.txt.

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class _Interaction(nn.Module):
    def forward(self, x):
        return F.gelu(sum(x))


class _LogPower(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x**2, dim=3), 1e-4, 1e4))


class _ProjectedLinear(nn.Linear):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        # Preserve the author's non-differentiable forward weight projection.
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class _Stem(nn.Module):
    def __init__(
        self,
        num_electrodes,
        num_filters,
        num_bands,
        temporal_kernel,
        patch_size,
        dropout,
        log_power,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.patch_size = patch_size
        self.log_power = log_power
        # Native module names are retained for strict author checkpoint loading.
        self.sconv = _ConvBlock(
            num_electrodes * num_bands, num_filters * num_bands, 1, groups=num_bands
        )
        self.tconv = nn.ModuleList()
        for _ in range(num_bands):
            self.tconv.append(
                _ConvBlock(
                    num_filters,
                    num_filters,
                    temporal_kernel,
                    groups=num_filters,
                    padding=temporal_kernel // 2,
                )
            )
            temporal_kernel //= 2
        self.interFre = _Interaction()
        if log_power:
            self.power = _LogPower()
        else:
            self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, _, length = x.shape
        bands = torch.split(self.sconv(x), self.num_filters, dim=1)
        x = self.interFre([conv(band) for conv, band in zip(self.tconv, bands)])
        if self.log_power:
            x = self.power(
                x.reshape(
                    batch_size,
                    self.num_filters,
                    length // self.patch_size,
                    self.patch_size,
                )
            )
        else:
            x = self.downSampling(x)
        return self.dp(x)


class _IFNet(nn.Module):
    def __init__(
        self,
        num_electrodes,
        chunk_size,
        num_classes,
        num_filters,
        num_bands,
        temporal_kernel,
        patch_size,
        dropout,
        log_power,
        max_norm=0.5,
    ):
        super().__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('num_filters', num_filters),
            ('num_bands', num_bands),
            ('temporal_kernel', temporal_kernel),
            ('patch_size', patch_size),
        ]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        kernel = temporal_kernel
        for _ in range(num_bands):
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(
                    'Successive halved temporal kernels must be positive and odd.'
                )
            kernel //= 2
        if chunk_size < patch_size or (log_power and chunk_size % patch_size):
            raise ValueError(
                'chunk_size must cover a patch; IFNetV2 also needs an exact multiple.'
            )
        if (
            not isinstance(dropout, Real)
            or isinstance(dropout, bool)
            or not math.isfinite(dropout)
            or not 0 <= dropout <= 1
        ):
            raise ValueError('dropout must be between zero and one.')
        if (
            not isinstance(max_norm, Real)
            or isinstance(max_norm, bool)
            or not math.isfinite(max_norm)
            or max_norm <= 0
        ):
            raise ValueError('max_norm must be positive and finite.')
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_bands = num_bands
        self.temporal_kernel = temporal_kernel
        self.patch_size = patch_size
        self.dropout = dropout
        self.stem = _Stem(
            num_electrodes,
            num_filters,
            num_bands,
            temporal_kernel,
            patch_size,
            dropout,
            log_power,
        )
        width = num_filters * (chunk_size // patch_size)
        if log_power:
            self.fc = nn.Sequential(
                _ProjectedLinear(width, num_classes, max_norm=max_norm)
            )
        else:
            self.fc = nn.Sequential(nn.Linear(width, num_classes))
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Classify prefiltered sub-bands with shape
        :obj:`[batch, bands, electrodes, time]` or
        :obj:`[batch, bands * electrodes, time]` in band-major order.

        Returns:
            torch.Tensor: Logits with shape :obj:`[batch, num_classes]`.
        """
        if x.ndim == 4:
            if x.shape[1] != self.num_bands or x.shape[2] != self.num_electrodes:
                raise ValueError(
                    'Expected the configured number of bands and electrodes.'
                )
            x = x.reshape(x.shape[0], self.num_bands * self.num_electrodes, x.shape[-1])
        if (
            x.ndim != 3
            or x.shape[1] != self.num_bands * self.num_electrodes
            or x.shape[2] != self.chunk_size
        ):
            raise ValueError(
                'Expected prefiltered [batch, bands, electrodes, time] or '
                '[batch, bands * electrodes, time] matching the constructor.'
            )
        return self.fc(self.stem(x).flatten(1))


class IFNet(_IFNet):
    r"""Author IFNet with average temporal pooling.

    - Paper: Wang J, Yao L, Wang Y. IFNet: An Interactive Frequency
      Convolutional Neural Network for Enhancing Motor Imagery Decoding
      from EEG. IEEE TNSRE, 2023.
    - URL: https://doi.org/10.1109/TNSRE.2023.3257319
    - Related Project: https://github.com/Jiaheng-Wang/IFNet

    Uses grouped spatial projection, band-specific depthwise temporal
    convolutions, additive cross-band GELU interaction, average pooling and
    a dense classifier. Provide prefiltered sub-bands; raw EEG is not
    automatically filtered or repeated. Returns logits for cross entropy.
    See :class:`IFNetV2` for the author's log-power variant.

    .. code-block:: python

        from torcheeg.models import IFNet
        import torch

        model = IFNet(num_electrodes=22, num_bands=2, chunk_size=750)
        logits = model(torch.randn(3, 2, 22, 750))

    Args:
        num_electrodes (int): Electrodes per frequency band. (default: :obj:`22`)
        chunk_size (int): Time samples per epoch. (default: :obj:`750`)
        num_classes (int): Output classes. (default: :obj:`4`)
        num_filters (int): Spatial features per band. (default: :obj:`64`)
        num_bands (int): Number of prefiltered frequency bands. (default: :obj:`2`)
        temporal_kernel (int): Largest temporal kernel. Repeated integer
            halving must yield a positive odd kernel for each band.
            (default: :obj:`63`)
        patch_size (int): Average-pooling window and stride. Incomplete final
            patches are discarded. (default: :obj:`125`)
        dropout (float): Dropout probability. (default: :obj:`0.5`)
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 750,
        num_classes: int = 4,
        num_filters: int = 64,
        num_bands: int = 2,
        temporal_kernel: int = 63,
        patch_size: int = 125,
        dropout: float = 0.5,
    ):
        super().__init__(
            num_electrodes,
            chunk_size,
            num_classes,
            num_filters,
            num_bands,
            temporal_kernel,
            patch_size,
            dropout,
            log_power=False,
        )


class IFNetV2(_IFNet):
    r"""Author IFNet V2 with log-power pooling and projected readout weights.

    Uses the complete author IFNetV2.py network. Input is prefiltered bands
    in the same layouts as :class:`IFNet`. Log-power clamps mean squared
    activations to :math:`[10^{-4}, 10^4]` before logarithms. The classifier
    projects its weights before each forward without differentiating through
    the projection, as in the author implementation. This differs from a
    differentiable parametrization. The epoch must contain complete patches.

    - Paper: https://doi.org/10.1109/TNSRE.2023.3257319
    - Related Project: https://github.com/Jiaheng-Wang/IFNet

    .. code-block:: python

        from torcheeg.models import IFNetV2
        import torch

        model = IFNetV2(num_electrodes=22, num_bands=2, chunk_size=750)
        logits = model(torch.randn(3, 2, 22, 750))

    Args:
        num_electrodes (int): Electrodes per band. (default: :obj:`22`)
        chunk_size (int): Epoch samples, divisible by patch_size.
            (default: :obj:`750`)
        num_classes (int): Output classes. (default: :obj:`4`)
        num_filters (int): Spatial features per band. (default: :obj:`64`)
        num_bands (int): Prefiltered frequency bands. (default: :obj:`2`)
        temporal_kernel (int): Largest kernel; all successive halved kernels
            must remain positive and odd. (default: :obj:`63`)
        patch_size (int): Samples per log-power patch. (default: :obj:`125`)
        dropout (float): Dropout probability. (default: :obj:`0.5`)
        max_norm (float): Classifier row L2 norm bound. (default: :obj:`0.5`)
    """

    def __init__(
        self,
        num_electrodes: int = 22,
        chunk_size: int = 750,
        num_classes: int = 4,
        num_filters: int = 64,
        num_bands: int = 2,
        temporal_kernel: int = 63,
        patch_size: int = 125,
        dropout: float = 0.5,
        max_norm: float = 0.5,
    ):
        super().__init__(
            num_electrodes,
            chunk_size,
            num_classes,
            num_filters,
            num_bands,
            temporal_kernel,
            patch_size,
            dropout,
            log_power=True,
            max_norm=max_norm,
        )
