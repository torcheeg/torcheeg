import torch.nn as nn
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                                               [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Conv2dBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        ConvLayer = Conv2dSamePadding(
            in_channels, out_channels, kernel_size, stride, bias=False, **kwargs
        )
        super(Conv2dBnReLU, self).__init__(
            ConvLayer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class TinySleepNet(nn.Module):
    r'''
    An efficient deep learning model for automatic sleep stage scoring based on raw single-channel EEG. For more details, please refer to the following information.

    - Paper: A. Supratak and Y. Guo, "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG," Annu Int Conf IEEE Eng Med Biol Soc, vol. 2020, pp. 641-644, Jul 2020.
    - URL: https://ieeexplore.ieee.org/document/9176741
    - Related Project: https://github.com/akaraspt/tinysleepnet

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import TinySleepNet

        model = TinySleepNet(num_classes=5, 
                            chunk_size=3000,
                            num_electrodes=1)

        # batch_size, 1, chunk_size, num_electrodes
        x = torch.randn(32, 1, 3000, 1)
        model(x)

    Args:
        num_classes (int): The number of sleep stages to classify. (default: :obj:`2`)
        chunk_size (int): Number of data points in each EEG segment. (default: :obj:`3000`)
        num_electrodes (int): The number of EEG channels. (default: :obj:`1`)
        dropout (float): Dropout rate for regularization. (default: :obj:`0.5`)
        n_filters_1 (int): Number of filters in the first convolution layer. (default: :obj:`128`)
        filter_size_1 (int): Filter size for the first convolution layer. (default: :obj:`50`)
        filter_stride_1 (int): Stride for the first convolution layer. (default: :obj:`6`)
        pool_size_1 (int): Pooling size after first convolution layer. (default: :obj:`8`)
        pool_stride_1 (int): Pooling stride after first convolution layer. (default: :obj:`8`)
        n_filters_1x3 (int): Number of filters in residual blocks. (default: :obj:`128`)
        filter_size_1x3 (int): Filter size in residual blocks. (default: :obj:`8`)
        pool_size_2 (int): Final pooling size. (default: :obj:`4`)
        pool_stride_2 (int): Final pooling stride. (default: :obj:`4`)
    '''

    def __init__(
        self,
        num_classes: int = 2,
        chunk_size: int = 3000,
        num_electrodes: int = 1,
        dropout: float = 0.5,
        n_filters_1: int = 128,
        filter_size_1: int = 50,
        filter_stride_1: int = 6,
        pool_size_1: int = 8,
        pool_stride_1: int = 8,
        n_filters_1x3: int = 128,
        filter_size_1x3: int = 8,
        pool_size_2: int = 4,
        pool_stride_2: int = 4,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1,
                         num_electrodes), (filter_stride_1, num_electrodes)),
            nn.MaxPool2d((pool_size_1, 1), (pool_stride_1, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_2, 1), (pool_stride_2, 1)),
            nn.Dropout(dropout)
        )

        outlen_conv1 = chunk_size // filter_stride_1 // pool_stride_1 // pool_stride_2
        outlen_conv = outlen_conv1*n_filters_1x3

        self.feature_dim = outlen_conv
        self.classifier = nn.Linear(
            outlen_conv, num_classes) if num_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
