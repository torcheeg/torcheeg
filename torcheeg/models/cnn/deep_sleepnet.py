from functools import reduce
from operator import __add__

import torch
import torch.nn as nn


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


class DeepSleepNet(nn.Module):
    r'''
    A deep learning model for automatic sleep stage scoring based on raw single-channel EEG. For more details, please refer to the following information.

    - Paper: Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017). DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), 1998-2008.
    - URL: https://ieeexplore.ieee.org/abstract/document/7961240
    - Related Project: https://github.com/akaraspt/deepsleepnet

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import DeepSleepNet

        model = DeepSleepNet(num_classes=5, 
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
        n_filters_1 (int): Number of filters in the first small-filter convolution path. (default: :obj:`64`)
        filter_size_1 (int): Filter size for the first convolution in small-filter path. (default: :obj:`50`)
        filter_stride_1 (int): Stride for the first convolution in small-filter path. (default: :obj:`6`)
        n_filters_2 (int): Number of filters in the first large-filter convolution path. (default: :obj:`64`)
        filter_size_2 (int): Filter size for the first convolution in large-filter path. (default: :obj:`400`)
        filter_stride_2 (int): Stride for the first convolution in large-filter path. (default: :obj:`50`)
        pool_size_11 (int): Pooling size after first small-filter convolution. (default: :obj:`8`)
        pool_stride_11 (int): Pooling stride after first small-filter convolution. (default: :obj:`8`)
        pool_size_21 (int): Pooling size after first large-filter convolution. (default: :obj:`4`)
        pool_stride_21 (int): Pooling stride after first large-filter convolution. (default: :obj:`4`)
        n_filters_1x3 (int): Number of filters in small-filter path's residual blocks. (default: :obj:`128`)
        filter_size_1x3 (int): Filter size in small-filter path's residual blocks. (default: :obj:`8`)
        n_filters_2x3 (int): Number of filters in large-filter path's residual blocks. (default: :obj:`128`)
        filter_size_2x3 (int): Filter size in large-filter path's residual blocks. (default: :obj:`6`)
        pool_size_12 (int): Final pooling size in small-filter path. (default: :obj:`4`)
        pool_stride_12 (int): Final pooling stride in small-filter path. (default: :obj:`4`)
        pool_size_22 (int): Final pooling size in large-filter path. (default: :obj:`2`)
        pool_stride_22 (int): Final pooling stride in large-filter path. (default: :obj:`2`)
    '''

    def __init__(
        self,
        num_classes: int = 2,
        chunk_size: int = 3000,
        num_electrodes: int = 1,
        dropout: float = 0.5,
        n_filters_1: int = 64,
        filter_size_1: int = 50,
        filter_stride_1: int = 6,
        n_filters_2: int = 64,
        filter_size_2: int = 400,
        filter_stride_2: int = 50,
        pool_size_11: int = 8,
        pool_stride_11: int = 8,
        pool_size_21: int = 4,
        pool_stride_21: int = 4,
        n_filters_1x3: int = 128,
        filter_size_1x3: int = 8,
        n_filters_2x3: int = 128,
        filter_size_2x3: int = 6,
        pool_size_12: int = 4,
        pool_stride_12: int = 4,
        pool_size_22: int = 2,
        pool_stride_22: int = 2
    ):
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes

        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, num_electrodes),
                         (filter_stride_1, num_electrodes)),
            nn.MaxPool2d((pool_size_11, 1), (pool_stride_11, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3,
                         (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_12, 1), (pool_stride_12, 1)),
        )
        self.conv2 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_2, (filter_size_2, 1),
                         (filter_stride_2, 1)),
            nn.MaxPool2d((pool_size_21, 1), (pool_stride_21, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_2,   n_filters_2x3,
                         (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3,
                         (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3,
                         (filter_size_2x3, 1), stride=1),
            nn.MaxPool2d((pool_size_22, 1), (pool_stride_22, 1)),
        )
        self.drop1 = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            self.feature_dim(), num_classes) if num_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(2, 1, self.chunk_size, self.num_electrodes)

            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x = torch.cat((x1, x2), dim=1)

        return x.shape[-1]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.drop1(x)
        x = self.classifier(x)
        return x
