import torch
import torch.nn as nn


class EEGDepthAttention(nn.Module):

    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1,
                              1,
                              kernel_size=(k, 1),
                              padding=(k // 2, 0),
                              bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        return y * self.C * x


class LMDA(nn.Module):
    r'''
    A lightweight multi-dimensional attention network (LMDA-Net). For more details, please refer to the following information.

    - Paper: Miao Z, Zhao M, Zhang X, et al. LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interfaces and interpretability[J]. NeuroImage, 2023, 276: 120209.
    - URL: https://www.sciencedirect.com/science/article/pii/S1053811923003609
    - Related Project: https://github.com/MiaoZhengQing/LMDA-Code

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import LMDA

        model = LMDA(chunk_size=1750,
                    num_electrodes=22,
                    num_classes=4,
                    depth=9,
                    kernel=75)

        # batch_size, num_electrodes, n_electrodes, chunk_size
        x = torch.randn(64, 1, 22, 1750)
        model(x)

    Args:
        num_electrodes (int): The number of electrodes, i.e., number of channels. (default: :obj:`22`)
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`1750`)
        num_classes (int): The number of classes to predict. (default: :obj:`4`)
        depth (int): The depth of the channel attention mechanism. (default: :obj:`9`)
        kernel (int): The kernel size for temporal convolution. (default: :obj:`75`) 
        hid_channels_1 (int): The number of hidden channels in the first convolution block. (default: :obj:`24`)
        hid_channels_2 (int): The number of hidden channels in the second convolution block. (default: :obj:`9`)
        pool_size (int): The size of the average pooling layer. (default: :obj:`5`)
    '''
    def __init__(self,
                 num_electrodes: int = 22,
                 chunk_size: int = 1750,
                 num_classes: int = 4,
                 depth: int = 9,
                 kernel: int = 75,
                 hid_channels_1: int = 24,
                 hid_channels_2: int = 9,
                 pool_size: int = 5):
        
        super(LMDA, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1,
                                                       num_electrodes),
                                           requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth,
                      hid_channels_1,
                      kernel_size=(1, 1),
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(hid_channels_1),
            nn.Conv2d(hid_channels_1,
                      hid_channels_1,
                      kernel_size=(1, kernel),
                      groups=hid_channels_1,
                      bias=False),
            nn.BatchNorm2d(hid_channels_1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(hid_channels_1,
                      hid_channels_2,
                      kernel_size=(1, 1),
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(hid_channels_2),
            nn.Conv2d(hid_channels_2,
                      hid_channels_2,
                      kernel_size=(num_electrodes, 1),
                      groups=hid_channels_2,
                      bias=False),
            nn.BatchNorm2d(hid_channels_2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, pool_size)),
            nn.Dropout(p=0.65),
        )

        out = torch.ones((1, 1, num_electrodes, chunk_size))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)

        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(
            n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        x_time = self.time_conv(x)
        x_time = self.depthAttention(x_time)

        x = self.chanel_conv(x_time)
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls
