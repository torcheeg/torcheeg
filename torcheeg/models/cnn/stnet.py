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
        model = STNet(num_classes=2, chunk_size=128, grid_size=(9, 9), dropout=0.2)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`128`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.2`)
    '''
    def __init__(self,
                 chunk_size: int = 128,
                 grid_size: Tuple[int, int] = (9, 9),
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super(STNet, self).__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.grid_size = grid_size

        self.layer1 = nn.Conv2d(chunk_size, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer3 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer4 = SeparableConv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.layer5 = InceptionConv2d(32, 16)

        self.drop_selu = nn.Sequential(nn.Dropout(p=dropout), nn.SELU())

        self.lin1 = nn.Linear(self.feature_dim, 1024, bias=True)
        self.lin2 = nn.Linear(1024, num_classes, bias=True)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.chunk_size, *self.grid_size)

            mock_eeg = self.layer1(mock_eeg)
            mock_eeg = self.drop_selu(mock_eeg)
            mock_eeg = self.layer2(mock_eeg)
            mock_eeg = self.drop_selu(mock_eeg)
            mock_eeg = self.layer3(mock_eeg)
            mock_eeg = self.drop_selu(mock_eeg)
            mock_eeg = self.layer4(mock_eeg)
            mock_eeg = self.drop_selu(mock_eeg)
            mock_eeg = self.layer5(mock_eeg)
            mock_eeg = self.drop_selu(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 128, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`128` corresponds to :obj:`chunk_size`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

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
        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.drop_selu(x)
        x = self.lin2(x)
        return x
