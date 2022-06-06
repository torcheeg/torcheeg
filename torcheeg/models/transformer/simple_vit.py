from typing import Tuple

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches: torch.Tensor,
                     temperature: int = 10000,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hid_channels),
            nn.GELU(),
            nn.Linear(hid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, hid_channels: int, heads: int = 8, head_channels: int = 64):
        super().__init__()
        inner_channels = head_channels * heads
        self.heads = heads
        self.scale = head_channels**-0.5
        self.norm = nn.LayerNorm(hid_channels)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(hid_channels, inner_channels * 3, bias=False)
        self.to_out = nn.Linear(inner_channels, hid_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, hid_channels: int, depth: int, heads: int, head_channels: int, mlp_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(hid_channels, heads=heads, head_channels=head_channels),
                    FeedForward(hid_channels, mlp_channels)
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    r'''
    A Simple and Effective Vision Transformer (SimpleViT). The authors of Vision Transformer (ViT) present a few minor modifications and dramatically improve the performance of plain ViT models. For more details, please refer to the following information. 

    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.

    - Paper: Beyer L, Zhai X, Kolesnikov A. Better plain ViT baselines for ImageNet-1k[J]. arXiv preprint arXiv:2205.01580, 2022.
    - URL: https://arxiv.org/abs/2205.01580
    - Related Project: https://github.com/lucidrains/vit-pytorch

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BandDifferentialEntropy({
                            "delta": [1, 4],
                            "theta": [4, 8],
                            "alpha": [8, 14],
                            "beta": [14, 31],
                            "gamma": [31, 49]
                        }),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = SimpleViT(hid_channels=32,
                          depth=3,
                          heads=4,
                          mlp_channels=64,
                          grid_size=(9, 9),
                          patch_size=3,
                          num_classes=2,
                          in_channels=5,
                          head_channels=64)

    Args:
        in_channels (int): The feature dimension of each electrode. (defualt: :obj:`5`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        patch_size (int): The size (resolution) of each input patch. (defualt: :obj:`3`)
        hid_channels (int): The feature dimension of embeded patch. (defualt: :obj:`32`)
        depth (int): The number of attention layers for each transformer block. (defualt: :obj:`3`)
        heads (int): The number of attention heads for each attention layer. (defualt: :obj:`4`)
        head_channels (int): The dimension of each attention head for each attention layer. (defualt: :obj:`8`)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (defualt: :obj:`64`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
    '''
    def __init__(self,
                 in_channels: int = 5,
                 grid_size: Tuple[int, int] = (9, 9),
                 patch_size: int = 3,
                 hid_channels: int = 32,
                 depth: int = 3,
                 heads: int = 4,
                 head_channels: int = 8,
                 mlp_channels: int = 64,
                 num_classes: int = 2):
        super().__init__()
        grid_height, grid_width = pair(grid_size)
        patch_height, patch_width = pair(patch_size)

        assert grid_height % patch_height == 0 and grid_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (grid_height // patch_height) * (grid_width // patch_width)
        patch_channels = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_channels, hid_channels),
        )

        self.transformer = Transformer(hid_channels, depth, heads, head_channels, mlp_channels)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(hid_channels), nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, h, w, dtype = *x.shape, x.dtype

        x = self.to_patch_embedding(x)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
