from typing import Tuple

import torch
import torch.nn as nn


class MTCNN(nn.Module):
    r'''
    Multi-Task Convolutional Neural Network (MT-CNN). For more details, please refer to the following information.

    - Paper: Rudakov E, Laurent L, Cousin V, et al. Multi-Task CNN model for emotion recognition from EEG Brain maps[C]//2021 4th International Conference on Bio-Engineering for Smart Technologies (BioSMART). IEEE, 2021: 1-4.
    - URL: https://ieeexplore.ieee.org/abstract/document/9677807
    - Related Project: https://github.com/dolphin-in-a-coma/multi-task-cnn-eeg-emotion/

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python
    
        DEAP_LOCATION_LIST = [['-', '-', 'AF3', 'FP1', '-', 'FP2', 'AF4', '-', '-'],
                              ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
                              ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
                              ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
                              ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
                              ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
                              ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
                              ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']]
        DEAP_CHANNEL_LOCATION_DICT = format_channel_location_dict(DEAP_CHANNEL_LIST, DEAP_LOCATION_LIST)

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.Concatenate([
                            transforms.BandDifferentialEntropy(),
                            transforms.BandPowerSpectralDensity()
                        ]),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = MTCNN(num_classes=2, in_channels=8, grid_size=(8, 9), dropout=0.2)

    Args:
        in_channels (int): The feature dimension of each electrode, i.e., :math:`N` in the paper. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(8, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.2`)
    '''
    def __init__(self,
                 in_channels: int = 8,
                 grid_size: Tuple[int, int] = (8, 9),
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super(MTCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.grid_size = grid_size

        self.block1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(64), nn.Dropout2d(dropout))
        self.block2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, padding=0, stride=1),
                                    nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(dropout))
        self.block3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, padding=0, stride=1),
                                    nn.ReLU(), nn.BatchNorm2d(256), nn.Dropout2d(dropout))
        self.block4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.Dropout2d(dropout))
        self.lin1 = nn.Sequential(nn.Linear(self.feature_dim, 512), nn.ReLU())
        self.lin1_bn = nn.Sequential(nn.BatchNorm1d(1), nn.Dropout(dropout))
        self.lin_v = nn.Linear(512, num_classes)
        self.lin_a = nn.Linear(512, num_classes)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)
            mock_eeg = self.block4(mock_eeg)

            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 8, 8, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`8` corresponds to :obj:`in_channels`, and :obj:`(8, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)

        x = x.transpose(-1, -2)
        x = self.lin1_bn(x)
        x = x.transpose(-1, -2)

        return self.lin_v(x), self.lin_a(x)
