import math
import warnings

import torch
import torch.nn as nn


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


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter

        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[
            0]
        median_energy = median_energy.view(B, 1)

        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float(
        ) - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class TSLANetLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ASB = AdaptiveSpectralBlock(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ICB = ICB(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + \
            self.drop_path(self.ICB(self.norm2(self.ASB(self.norm1(x)))))
        return x


class TSLANet(nn.Module):
    r'''
    A time series lightweight adaptive network for EEG classification. For more details, please refer to the following information.

    - Paper: Eldele E, Ragab M, Chen Z, et al. TSLANet: Rethinking Transformers for Time Series Representation Learning[C]//Forty-first International Conference on Machine Learning.
    - URL: https://openreview.net/pdf?id=CGR3vpX63X
    - Related Project: https://github.com/emadeldeen24/TSLANet

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import TSLANet

        model = TSLANet(num_classes=5,
                       chunk_size=3000,
                       patch_size=200,
                       num_electrodes=1)

        # batch_size, num_electrodes, chunk_size
        x = torch.randn(32, 1, 3000)
        model(x)

    Args:
        chunk_size (int): Number of data points in each EEG segment. (default: :obj:`3000`)
        patch_size (int): Size of each patch the input sequence is divided into. (default: :obj:`200`)
        num_electrodes (int): The number of EEG channels. (default: :obj:`6`)
        emb_dim (int): Dimension of the embedding space. (default: :obj:`128`)
        dropout_rate (float): Dropout rate for regularization. (default: :obj:`0.15`)
        depth (int): Number of TSLANet layers in the network. (default: :obj:`2`)
        num_classes (int): The number of classes to classify. (default: :obj:`2`)
    '''

    def __init__(self,
                 chunk_size: int = 3000,
                 patch_size: int = 200,
                 num_electrodes: int = 1,
                 emb_dim: int = 128,
                 dropout_rate: float = 0.15,
                 depth: int = 2,
                 num_classes: int = 2):
        super().__init__()
        self.emb_dim = emb_dim

        self.patch_embed = PatchEmbed(
            seq_len=chunk_size, patch_size=patch_size,
            in_chans=num_electrodes, embed_dim=emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.input_layer = nn.Linear(patch_size, emb_dim)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]

        self.tsla_blocks = nn.ModuleList([
            TSLANetLayer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i])
            for i in range(depth)]
        )

        self.head = nn.Linear(emb_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x
