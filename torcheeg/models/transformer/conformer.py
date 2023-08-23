import torch.nn as nn
import math
import torch

import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self,
                 num_electrodes: int,
                 hid_channels: int = 40,
                 dropout: float = 0.5):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, hid_channels, (1, 25), (1, 1)),
            nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1), (1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 75), (1, 15)
            ),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, (1, 1), stride=(
                1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_channels: int, heads: int, dropout: float):
        super().__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x),
                            "b n (h d) -> b h n d",
                            h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.hid_channels**(1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 hid_channels: int,
                 expansion: int = 4,
                 dropout: float = 0.):
        super().__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )


class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, hid_channels: int, heads: int, dropout: float,
                 forward_expansion: int, forward_dropout: float):
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(hid_channels),
                              MultiHeadAttention(hid_channels, heads, dropout),
                              nn.Dropout(dropout))),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    FeedForwardBlock(hid_channels,
                                     expansion=forward_expansion,
                                     dropout=forward_dropout),
                    nn.Dropout(dropout))))


class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth: int,
                 hid_channels: int,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5):
        super().__init__(*[
            TransformerEncoderBlock(hid_channels=hid_channels,
                                    heads=heads,
                                    dropout=dropout,
                                    forward_expansion=forward_expansion,
                                    forward_dropout=forward_dropout)
            for _ in range(depth)
        ])


class ClassificationHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hid_channels: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, hid_channels * 8),
                                nn.ELU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels * 8, hid_channels),
                                nn.ELU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conformer(nn.Module):
    r'''
    The EEG Conformer model is based on the paper "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization". For more details, please refer to the following information. 

    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.

    - Paper: Song Y, Zheng Q, Liu B, et al. EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2022.
    - URL: https://ieeexplore.ieee.org/document/9991178
    - Related Project: https://github.com/eeyhsong/EEG-Conformer

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              offline_transform=transforms.Compose([
                                  transforms.MinMaxNormalize(axis=-1),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
        model = Conformer(num_electrodes=62,
                          sampling_rate=200,
                          hid_channels=40,
                          depth=6,
                          heads=10,
                          dropout=0.5,
                          forward_expansion=4,
                          forward_dropout=0.5,
                          num_classes=2)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`62`)
        sampling_rate (int): The sampling rate of EEG signals. (default: :obj:`200`)
        hid_channels (int): The feature dimension of embeded patch. (default: :obj:`40`)
        depth (int): The number of attention layers for each transformer block. (default: :obj:`6`)
        heads (int): The number of attention heads for each attention layer. (default: :obj:`10`)
        dropout (float): The dropout rate of the attention layer. (default: :obj:`0.5`)
        forward_expansion (int): The expansion factor of the feedforward layer. (default: :obj:`4`)
        forward_dropout (float): The dropout rate of the feedforward layer. (default: :obj:`0.5`)
        num_classes (int): The number of classes. (default: :obj:`2`)
    '''
    def __init__(self,
                 num_electrodes: int = 62,
                 sampling_rate: int = 200,
                 embed_dropout: float = 0.5,
                 hid_channels: int = 40,
                 depth: int = 6,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5,
                 cls_channels: int = 32,
                 cls_dropout: float = 0.5,
                 num_classes: int = 2):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.sampling_rate = sampling_rate
        self.embed_dropout = embed_dropout
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        self.cls_channels = cls_channels
        self.cls_dropout = cls_dropout
        self.num_classes = num_classes

        self.embd = PatchEmbedding(num_electrodes, hid_channels, embed_dropout)
        self.encoder = TransformerEncoder(depth,
                                          hid_channels,
                                          heads=heads,
                                          dropout=dropout,
                                          forward_expansion=forward_expansion,
                                          forward_dropout=forward_dropout)
        self.cls = ClassificationHead(in_channels=self.feature_dim(),
                                      num_classes=num_classes,
                                      hid_channels=cls_channels,
                                      dropout=cls_dropout)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes,
                                   self.sampling_rate)

            mock_eeg = self.embd(mock_eeg)
            mock_eeg = self.encoder(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.cls(x)
        return x