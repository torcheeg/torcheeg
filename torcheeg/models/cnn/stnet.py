from typing import Tuple

import torch
import torch.nn as nn


class InceptionConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=bias)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv5x5(x) + self.conv3x3(x) + self.conv1x1(x)


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True):
        super().__init__()
        self.depth = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=in_channels,
                               bias=bias)
        self.point = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        return x


class STNet(nn.Module):
    r'''
    Spatio-temporal Network (STNet). For more details, please refer to the following information.

    - Paper: Zhang Z, Zhong S, Liu Y. GANSER: A Self-supervised Data Augmentation Framework for EEG-based Emotion Recognition[J]. IEEE Transactions on Affective Computing, 2022.
    - URL: https://ieeexplore.ieee.org/abstract/document/9763358/
    - Related Project: https://github.com/tczhangzhi/GANSER-PyTorch

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = STNet(num_classes=2, in_channels=4, grid_size=(9, 9), dropout=0.2)

    Args:
        in_channels (int): The dimension of each electrode. (defualt: :obj:`128`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (defualt: :obj:`0.2`)
    '''
    def __init__(self,
                 in_channels: int = 128,
                 grid_size: Tuple[int, int] = (9, 9),
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super(STNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.grid_size = grid_size

        self.layer1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer3 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer4 = SeparableConv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer5 = InceptionConv2d(32, 16)

        self.drop_selu = nn.Sequential(nn.Dropout(p=dropout), nn.SELU())

        self.lin1 = nn.Linear(grid_size[0] * grid_size[1] * 16, 1024, bias=True)
        self.lin2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 128, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`128` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.layer1(x)
        x = self.drop_selu(x)
        x = self.layer2(x)
        x = self.drop_selu(x)
        x = self.layer3(x)
        x = self.drop_selu(x)
        x = self.layer4(x)
        x = self.drop_selu(x)
        x = self.layer5(x)
        x = self.drop_selu(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.drop_selu(x)
        x = self.lin2(x)
        return x
