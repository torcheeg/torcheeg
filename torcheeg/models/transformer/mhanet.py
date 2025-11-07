import torch
import torch.nn as nn
from einops import rearrange


class MultiscaleTemporalLayer(nn.Module):
    """Multi-scale temporal convolution layer.

    Args:
        seq_len (int): The sequence length.
        kernel_size (int): The kernel size for convolution.
    """

    def __init__(self, seq_len: int, kernel_size: int):
        super(MultiscaleTemporalLayer, self).__init__()

        self.multiscale_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding='same'
        )
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(seq_len)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multiscale_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class MultiscaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention module.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        chunk_size (int): The sampling rate of EEG signals.
    """

    def __init__(self, num_electrodes: int, chunk_size: int):
        super(MultiscaleTemporalAttention, self).__init__()

        self.spatio_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(num_electrodes, 1)
        )
        self.up_channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.project_out = nn.Conv2d(
            in_channels=1,
            out_channels=num_electrodes,
            kernel_size=1,
            stride=1
        )

        self.multi_temporal_k_2 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=2)
        self.multi_temporal_k_4 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=4)
        self.multi_temporal_k_6 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)
        x = self.spatio_conv(x)
        x = self.up_channel_conv(x.squeeze(2))

        x, y, z = x.chunk(3, dim=1)

        x_attn = self.multi_temporal_k_2(x)
        y_attn = self.multi_temporal_k_4(y)
        z_attn = self.multi_temporal_k_6(z)

        out = x_attn * x + y_attn * y + z_attn * z
        out = out.view(batch_size, 1, 1, -1)
        out = self.project_out(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention module with multi-scale temporal attention.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        chunk_size (int): The sampling rate of EEG signals.
        dim (int): The dimension of channels.
        num_heads (int): The number of attention heads.
        bias (bool): Whether to use bias in convolution layers.
    """

    def __init__(self,
                 num_electrodes: int,
                 chunk_size: int,
                 num_heads: int,
                 bias: bool = False):
        super(ChannelAttention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(
            num_electrodes, num_electrodes * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            num_electrodes * 3,
            num_electrodes * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=num_electrodes * 3,
            bias=bias
        )
        self.project_out = nn.Conv2d(
            num_electrodes, num_electrodes, kernel_size=1, bias=bias)

        self.multiscale_temporal_attention = MultiscaleTemporalAttention(
            num_electrodes,
            chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        v = self.multiscale_temporal_attention(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class MultiscaleGlobalAttention(nn.Module):
    """Multi-scale global attention module with dilated convolutions."""

    def __init__(self):
        super(MultiscaleGlobalAttention, self).__init__()

        self.down_channel = nn.Conv2d(3, 1, 1, 1, 0)
        self.norm = nn.BatchNorm2d(1)
        self.dilation_rate = 3

        self.conv_0 = nn.Conv2d(1, 1, 3, padding='same', dilation=1)
        self.conv_1 = nn.Conv2d(1, 1, 5, padding='same', dilation=2)
        self.conv_2 = nn.Conv2d(1, 1, 7, padding='same',
                                dilation=self.dilation_rate)

        self.up_channel = nn.Sequential(
            nn.Conv2d(1, 3, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.norm(x)
        x = self.up_channel(x)
        y = x.clone()

        y1, y2, y3 = torch.chunk(y, 3, dim=1)

        attn_0 = self.conv_0(y1) * y1
        attn_1 = self.conv_1(y2) * y2
        attn_2 = self.conv_2(y3) * y3

        attn = torch.cat([attn_0, attn_1, attn_2], dim=1)
        out = x * attn
        out = self.down_channel(out) + shortcut

        return out


class SpatiotemporalConvolution(nn.Module):
    """Spatiotemporal convolution module.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        chunk_size (int): The sampling rate of EEG signals.
    """

    def __init__(self, num_electrodes: int, chunk_size: int):
        super(SpatiotemporalConvolution, self).__init__()

        self.temporal_convolution = nn.Sequential(
            nn.Conv2d(1, 5, (1, 2), stride=1),
            nn.BatchNorm2d(5),
            nn.ELU()
        )

        self.spatio_convolution = nn.Sequential(
            nn.Conv2d(5, 5, (num_electrodes, 1), stride=1),
            nn.BatchNorm2d(5),
            nn.ELU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_convolution(x)
        x = self.spatio_convolution(x)
        x = self.pool(x)
        return x


class MHANet(nn.Module):
    r'''
    The MHANet model is based on the paper "MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection". For more details, please refer to the following information.

    - Paper: Li L, Fan C, Zhang H, et al. MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection[J]. International Joint Conference on Artificial Intelligence, 2025.
    - URL: https://arxiv.org/abs/2505.15364
    - Related Project: https://github.com/fchest/MHANet

    Below is a recommended suite for use in auditory attention detection tasks:

    .. code-block:: python

        from torcheeg.models import MHANet
        from torcheeg.datasets import DTUDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        dataset = DTUDataset(root_path='./DATA_preproc',
                              offline_transform=transforms.Compose([
                                  transforms.MinMaxNormalize(axis=-1),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('attended_speaker'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]))

        model = MHANet(num_electrodes=64,
                       chunk_size=64,
                       num_heads=16,
                       bias=False,
                       num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`64`)
        chunk_size (int): The sampling rate of EEG signals. (default: :obj:`64`)
        num_heads (int): The number of attention heads. (default: :obj:`16`)
        bias (bool): Whether to use bias in convolution layers. (default: :obj:`False`)
        num_classes (int): The number of classes. (default: :obj:`2`)
    '''

    def __init__(self,
                 num_electrodes: int = 64,
                 chunk_size: int = 64,
                 num_heads: int = 16,
                 bias: bool = False,
                 num_classes: int = 2):
        super(MHANet, self).__init__()

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.bias = bias
        self.num_classes = num_classes

        self.channel_attention = ChannelAttention(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            num_heads=num_heads,
            bias=bias
        )

        self.multiscale_global_attention = MultiscaleGlobalAttention()
        self.spatiotemporal_convolution = SpatiotemporalConvolution(
            num_electrodes,
            chunk_size
        )

        self.flatten = nn.Flatten()
        self.out = nn.Linear(5, num_classes)

    def feature_dim(self) -> int:
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            x = mock_eeg.permute(0, 2, 1, 3)
            x = self.channel_attention(x)
            x = x.permute(0, 2, 1, 3)
            x = self.multiscale_global_attention(x)
            x = self.spatiotemporal_convolution(x)
            x = self.flatten(x)

            return x.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 64, 64]`. Here, :obj:`n` corresponds to the batch size, the first :obj:`64` corresponds to :obj:`num_electrodes`, and the second :obj:`64` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[size of batch, number of classes]: The predicted probability that the samples belong to the classes.
        '''
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1, 3)
        x = self.channel_attention(x)
        x = x.permute(0, 2, 1, 3)
        x = self.multiscale_global_attention(x)
        x = self.spatiotemporal_convolution(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
