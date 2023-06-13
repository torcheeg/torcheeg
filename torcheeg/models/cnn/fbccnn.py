from typing import Tuple

import torch
import torch.nn as nn


class FBCCNN(nn.Module):
    r'''
    Frequency Band Correlation Convolutional Neural Network (FBCCNN). For more details, please refer to the following information.

    - Paper: Pan B, Zheng W. Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network[J]. Computational and Mathematical Methods in Medicine, 2021.
    - URL: https://www.hindawi.com/journals/cmmm/2021/2520394/

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.BandPowerSpectralDensity(),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = FBCCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    Args:
        in_channels (int): The feature dimension of each electrode, i.e., :math:`N` in the paper. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2):
        super(FBCCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.grid_size = grid_size

        self.block1 = nn.Sequential(nn.Conv2d(in_channels, 12, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(12))
        self.block2 = nn.Sequential(nn.Conv2d(12, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(64))
        self.block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(128))
        self.block5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(256))
        self.block6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(128))
        self.block7 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.lin1 = nn.Sequential(nn.Linear(grid_size[0] * grid_size[1] * 32, 512), nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.lin3 = nn.Linear(128, num_classes)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)
            mock_eeg = self.block4(mock_eeg)
            mock_eeg = self.block5(mock_eeg)
            mock_eeg = self.block6(mock_eeg)
            mock_eeg = self.block7(mock_eeg)

            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x
