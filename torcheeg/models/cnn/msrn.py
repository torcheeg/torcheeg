from typing import Tuple

import torch
import torch.nn as nn


class MSRN(torch.nn.Module):
    r'''
    Multi-scale Residual Network (MSRN). For more details, please refer to the following information.

    - Paper: Li J, Hua H, Xu Z, et al. Cross-subject EEG emotion recognition combined with connectivity features and meta-transfer learning[J]. Computers in Biology and Medicine, 2022, 145: 105519.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0010482522003110
    - Related Project: https://github.com/ljy-scut/MTL-MSRN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BandSignal(),
                        transforms.PearsonCorrelation()
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = MSRN(num_classes=2, in_channels=4)

    Args:
        in_channels (int): The feature dimension of each electrode. (defualt: :obj:`4`)
        hid_channels (int): The number of kernels in the first convolutional layers. (defualt: :obj:`64`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (defualt: :obj:`0.25`)
    '''
    def __init__(self,
                 in_channels: int = 4,
                 hid_channels: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.hid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.BatchNorm2d(self.hid_channels), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ms_module_conv3x3 = nn.Sequential(
            nn.Conv2d(self.hid_channels,
                      self.in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.BatchNorm2d(self.in_channels), nn.ReLU(),
            nn.Conv2d(self.in_channels,
                      self.hid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(self.hid_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.hid_channels * 2,
                      self.hid_channels * 4,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(self.hid_channels * 4),
            nn.ReLU())

        self.ms_module_conv5x5 = nn.Sequential(
            nn.Conv2d(self.hid_channels,
                      self.in_channels,
                      kernel_size=5,
                      stride=1,
                      padding=2), nn.BatchNorm2d(self.in_channels), nn.ReLU(),
            nn.Conv2d(self.in_channels,
                      self.hid_channels,
                      kernel_size=5,
                      stride=1,
                      padding=2), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels * 2,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.BatchNorm2d(self.hid_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.hid_channels * 2,
                      self.hid_channels * 4,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.BatchNorm2d(self.hid_channels * 4),
            nn.ReLU())

        self.ms_module_conv7x7 = nn.Sequential(
            nn.Conv2d(self.hid_channels,
                      self.in_channels,
                      kernel_size=7,
                      stride=1,
                      padding=3), nn.BatchNorm2d(self.in_channels), nn.ReLU(),
            nn.Conv2d(self.in_channels,
                      self.hid_channels,
                      kernel_size=7,
                      stride=1,
                      padding=3), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3), nn.BatchNorm2d(self.hid_channels), nn.ReLU(),
            nn.Conv2d(self.hid_channels,
                      self.hid_channels * 2,
                      kernel_size=7,
                      stride=2,
                      padding=3), nn.BatchNorm2d(self.hid_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.hid_channels * 2,
                      self.hid_channels * 4,
                      kernel_size=7,
                      stride=2,
                      padding=3), nn.BatchNorm2d(self.hid_channels * 4),
            nn.ReLU())

        self.pool2 = nn.AdaptiveAvgPool2d(output_size=1)
        self.cls = nn.Sequential(
            nn.Linear(self.hid_channels * 12, self.hid_channels * 12),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hid_channels * 12, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 32, 32]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(32, 32)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.conv1(x)
        x = self.pool1(x)

        x1 = self.ms_module_conv3x3(x)
        x2 = self.ms_module_conv5x5(x)
        x3 = self.ms_module_conv7x7(x)
        x1 = self.pool2(x1)
        x2 = self.pool2(x2)
        x3 = self.pool2(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.cls(x)
        return x