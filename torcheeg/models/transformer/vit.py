from typing import Tuple

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, in_channels: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hid_channels, in_channels),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 hid_channels: int,
                 heads: int = 8,
                 head_channels: int = 64,
                 dropout: float = 0.):
        super(Attention, self).__init__()
        inner_channels = head_channels * heads
        project_out = not (heads == 1 and head_channels == hid_channels)

        self.heads = heads
        self.scale = head_channels**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(hid_channels, inner_channels * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_channels, hid_channels),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 hid_channels: int,
                 depth: int,
                 heads: int,
                 head_channels: int,
                 mlp_channels: int,
                 dropout: float = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        hid_channels,
                        Attention(hid_channels,
                                  heads=heads,
                                  head_channels=head_channels,
                                  dropout=dropout)),
                    PreNorm(
                        hid_channels,
                        FeedForward(hid_channels, mlp_channels,
                                    dropout=dropout))
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    r'''
    The Vision Transformer. For more details, please refer to the following information. It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.

    - Paper: Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.
    - URL: https://arxiv.org/abs/2010.11929
    - Related Project: https://github.com/lucidrains/vit-pytorch

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.MinMaxNormalize(axis=-1),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = ViT(chunk_size=128,
                    grid_size=(9, 9),
                    t_patch_size=32,
                    num_classes=2)

    It can also be used for the analysis of features such as DE, PSD, etc:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BandDifferentialEntropy(sampling_rate=128),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = ViT(chunk_size=4,
                    grid_size=(9, 9),
                    t_patch_size=1,
                    num_classes=2)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`128`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        t_patch_size (int): The size of each input patch at the temporal (chunk size) dimension. (default: :obj:`32`)
        s_patch_size (tuple): The size (resolution) of each input patch at the spatial (grid size) dimension. (default: :obj:`(3, 3)`)
        hid_channels (int): The feature dimension of embeded patch. (default: :obj:`32`)
        depth (int): The number of attention layers for each transformer block. (default: :obj:`3`)
        heads (int): The number of attention heads for each attention layer. (default: :obj:`4`)
        head_channels (int): The dimension of each attention head for each attention layer. (default: :obj:`8`)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (default: :obj:`64`)
        num_classes (int): The number of classes to predict. (default: :obj:`0.0`)
        embed_dropout (float): Probability of an element to be zeroed in the dropout layers of the embedding layers. (default: :obj:`0.0`)
        dropout (float): Probability of an element to be zeroed in the dropout layers of the transformer layers. (default: :obj:`0.0`)
        pool_func (str): The pool function before the classifier, optionally including :obj:`cls` and :obj:`mean`, where :obj:`cls` represents selecting classification-related token and :obj:`mean` represents the average pooling. (default: :obj:`cls`)
    '''
    def __init__(self,
                 chunk_size: int = 128,
                 grid_size: Tuple[int, int] = (9, 9),
                 t_patch_size: int = 32,
                 s_patch_size: Tuple[int, int] = (3, 3),
                 hid_channels: int = 32,
                 depth: int = 3,
                 heads: int = 4,
                 head_channels: int = 64,
                 mlp_channels: int = 64,
                 num_classes: int = 2,
                 embed_dropout: float = 0.,
                 dropout: float = 0.,
                 pool_func: str = 'cls'):
        super(ViT, self).__init__()
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.t_patch_size = t_patch_size
        self.s_patch_size = s_patch_size
        self.dropout = dropout
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.head_channels = head_channels
        self.mlp_channels = mlp_channels
        self.pool_func = pool_func
        self.embed_dropout = embed_dropout
        self.num_classes = num_classes

        grid_height, grid_width = pair(grid_size)
        patch_height, patch_width = pair(s_patch_size)

        assert grid_height % patch_height == 0 and grid_width % patch_width == 0, f'EEG grid size {grid_size} must be divisible by the spatial patch size {s_patch_size}.'
        assert chunk_size % t_patch_size == 0, f'EEG chunk size {chunk_size} must be divisible by the temporal patch size {t_patch_size}.'

        num_patches = (chunk_size // t_patch_size) * (
            grid_height // patch_height) * (grid_width // patch_width)
        patch_channels = t_patch_size * patch_height * patch_width
        assert pool_func in {
            'cls', 'mean'
        }, 'pool_func must be either cls (cls token) or mean (mean pooling).'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b c h w (p1 p2 p0)',
                      p0=t_patch_size,
                      p1=patch_height,
                      p2=patch_width),
            nn.Linear(patch_channels, hid_channels),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hid_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_channels))
        self.dropout = nn.Dropout(embed_dropout)

        self.transformer = Transformer(hid_channels, depth, heads,
                                       head_channels, mlp_channels, dropout)

        self.pool_func = pool_func

        self.mlp_head = nn.Sequential(nn.LayerNorm(hid_channels),
                                      nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 128, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`128` corresponds to :obj:`chunk_size`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.to_patch_embedding(x)
        x = rearrange(x, 'b ... d -> b (...) d')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool_func == 'mean' else x[:, 0]

        return self.mlp_head(x)