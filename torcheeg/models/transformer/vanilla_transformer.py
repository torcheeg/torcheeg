from typing import Tuple

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


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
    def __init__(self,
                 hid_channels: int,
                 heads: int = 8,
                 head_channels: int = 64):
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
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, hid_channels: int, depth: int, heads: int,
                 head_channels: int, mlp_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(hid_channels,
                              heads=heads,
                              head_channels=head_channels),
                    FeedForward(hid_channels, mlp_channels)
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VanillaTransformer(nn.Module):
    r'''
    A vanilla version of the transformer adapted on EEG analysis. For more details, please refer to the following information. 

    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.

    - Paper: Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.
    - URL: https://arxiv.org/abs/1706.03762
    - Related Project: https://github.com/huggingface/transformers

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.To2d(),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = VanillaTransformer(chunk_size=128,
                            num_electrodes=32,
                            patch_size=32,
                            hid_channels=32,
                            depth=3,
                            heads=4,
                            head_channels=64,
                            mlp_channels=64,
                            num_classes=2)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`128`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        patch_size (tuple): The size (resolution) of each input patch. (default: :obj:`(3, 3)`)
        hid_channels (int): The feature dimension of embeded patch. (default: :obj:`32`)
        depth (int): The number of attention layers for each transformer block. (default: :obj:`3`)
        heads (int): The number of attention heads for each attention layer. (default: :obj:`4`)
        head_channels (int): The dimension of each attention head for each attention layer. (default: :obj:`8`)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (default: :obj:`64`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 num_electrodes: int = 32,
                 chunk_size: int = 128,
                 t_patch_size: int = 32,
                 hid_channels: int = 32,
                 depth: int = 3,
                 heads: int = 4,
                 head_channels: int = 8,
                 mlp_channels: int = 64,
                 num_classes: int = 2):
        super(VanillaTransformer, self).__init__()
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.t_patch_size = t_patch_size
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.head_channels = head_channels
        self.mlp_channels = mlp_channels
        self.num_classes = num_classes

        assert chunk_size % t_patch_size == 0, f'EEG chunk size {chunk_size} must be divisible by the patch size {t_patch_size}.'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p) -> b (c w) p', p=t_patch_size),
            nn.Linear(t_patch_size, hid_channels),
        )
        num_patches = num_electrodes * (chunk_size // t_patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hid_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_channels))

        self.transformer = Transformer(hid_channels, depth, heads,
                                       head_channels, mlp_channels)

        self.linear_head = nn.Sequential(nn.LayerNorm(hid_channels),
                                         nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`chunk_size` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.linear_head(x)