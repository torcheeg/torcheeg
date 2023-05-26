from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSTEmotionNet(nn.Module):
    r'''
    Spatial-Spectral-Temporal based Attention 3D Dense Network (SST-EmotionNet) for EEG emotion recognition. For more details, please refer to the following information.

    - Paper: Jia Z, Lin Y, Cai X, et al. Sst-emotionnet: Spatial-spectral-temporal based attention 3d dense network for eeg emotion recognition[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 2909-2917.
    - URL: https://dl.acm.org/doi/abs/10.1145/3394171.3413724
    - Related Project: https://github.com/ziyujia/SST-EmotionNet
    - Related Project: https://github.com/LexieLiu01/SST-Emotion-Net-Pytorch-Version-
    
    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python
    
        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BaselineRemoval(),
                        transforms.Concatenate([
                            transforms.Compose([
                                transforms.BandDifferentialEntropy(sampling_rate=128),
                                transforms.MeanStdNormalize()
                            ]),
                            transforms.Compose([
                                transforms.Downsample(num_points=32),
                                transforms.MinMaxNormalize()
                            ])
                        ]),
                        transforms.ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((16, 16))
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = SSTEmotionNet(temporal_in_channels=32, spectral_in_channels=4, grid_size=(16, 16), num_classes=2)

    Args:
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(16, 16)`)
        spectral_in_channels (int): How many 2D maps are stacked in the 3D spatial-spectral representation. (default: :obj:`5`)
        temporal_in_channels (int): How many 2D maps are stacked in the 3D spatial-temporal representation. (default: :obj:`25`)
        spectral_depth (int): The number of layers in spatial-spectral stream. (default: :obj:`16`)
        temporal_depth (int): The number of layers in spatial-temporal stream. (default: :obj:`22`)
        spectral_growth_rate (int): The growth rate of spatial-spectral stream. (default: :obj:`12`)
        temporal_growth_rate (int): The growth rate of spatial-temporal stream. (default: :obj:`24`)
        num_dense_block (int): The number of A3DBs to add to end (default: :obj:`3`)
        hid_channels (int): The basic hidden channels in the network blocks. (default: :obj:`50`)
        densenet_dropout (int): Probability of an element to be zeroed in the dropout layers from densenet blocks. (default: :obj:`0.0`)
        task_dropout (int): Probability of an element to be zeroed in the dropout layers from task-specific classification blocks. (default: :obj:`0.0`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 grid_size: Tuple[int, int] = (32, 32),
                 spectral_in_channels: int = 5,
                 temporal_in_channels: int = 25,
                 spectral_depth: int = 16,
                 temporal_depth: int = 22,
                 spectral_growth_rate: int = 12,
                 temporal_growth_rate: int = 24,
                 num_dense_block: int = 3,
                 hid_channels: int = 50,
                 densenet_dropout: float = 0.0,
                 task_dropout: float = 0.0,
                 num_classes: int = 3):
        super(SSTEmotionNet, self).__init__()
        self.grid_size = grid_size
        self.spectral_in_channels = spectral_in_channels
        self.temporal_in_channels = temporal_in_channels
        self.spectral_depth = spectral_depth
        self.spectral_growth_rate = spectral_growth_rate
        self.temporal_growth_rate = temporal_growth_rate
        self.num_dense_block = num_dense_block
        self.hid_channels = hid_channels
        self.densenet_dropout = densenet_dropout
        self.task_dropout = task_dropout
        self.num_classes = num_classes

        assert grid_size[0] >= 16 and grid_size[
            1] >= 16, 'The height and width of the grid must be greater than or equal to 16. Please upsample the EEG grid.'

        self.spatial_spectral = DenseNet3D(grid_size=grid_size,
                                           in_channels=spectral_in_channels,
                                           depth=spectral_depth,
                                           num_dense_block=num_dense_block,
                                           growth_rate=spectral_growth_rate,
                                           reduction=0.5,
                                           bottleneck=True,
                                           dropout=densenet_dropout)

        self.spatial_temporal = DenseNet3D(grid_size=grid_size,
                                           in_channels=temporal_in_channels,
                                           depth=temporal_depth,
                                           num_dense_block=num_dense_block,
                                           growth_rate=temporal_growth_rate,
                                           bottleneck=True,
                                           subsample_initial_block=True,
                                           dropout=densenet_dropout)

        layers = []
        spectral_out, temporal_out = self.get_feature_dims()

        layers.append(nn.Linear(spectral_out + temporal_out, hid_channels))
        layers.append(nn.Dropout(p=task_dropout))
        layers.append(nn.Linear(hid_channels, num_classes))

        self.layers = nn.ModuleList(layers)

    def get_feature_dims(self):
        mock_eeg_s = torch.randn(2, self.grid_size[0], self.grid_size[1],
                                 self.spectral_in_channels)
        mock_eeg_t = torch.randn(2, self.grid_size[0], self.grid_size[1],
                                 self.temporal_in_channels)

        spectral_output = self.spatial_spectral(mock_eeg_s)
        temporal_output = self.spatial_temporal(mock_eeg_t)

        return spectral_output.shape[1], temporal_output.shape[1]

    def forward(self, x: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 30, 16, 16]`. Here, :obj:`n` corresponds to the batch size, :obj:`36` corresponds to the sum of :obj:`spectral_in_channels` (e.g., 5) and :obj:`temporal_in_channels` (e.g., 25), and :obj:`(16, 16)` corresponds to :obj:`grid_size`. It is worth noting that the first :obj:`spectral_in_channels` channels should represent spectral information.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        assert x.shape[1] == (
            self.spectral_in_channels + self.temporal_in_channels
        ), f'The input number of channels is {x.shape[1]}, but the expected number of channels is the number of spectral channels {self.spectral_in_channels} plus the number of temporal channels {self.temporal_in_channels}.'

        spectral_input = x[:, :self.spectral_in_channels]
        temporal_input = x[:, self.spectral_in_channels:]

        spectral_input = spectral_input.permute(0, 2, 3, 1)
        temporal_input = temporal_input.permute(0, 2, 3, 1)

        spectral_output = self.spatial_spectral(spectral_input)
        temporal_output = self.spatial_temporal(temporal_input)
        output = torch.cat([spectral_output, temporal_output], dim=1)
        for layer in self.layers:
            output = layer(output)

        return output


class DenseNet3D(nn.Module):
    def __init__(
        self,
        grid_size,
        in_channels,
        depth=40,
        num_dense_block=3,
        growth_rate=12,
        bottleneck=False,
        reduction=0.0,
        dropout=None,
        subsample_initial_block=False,
    ):
        super(DenseNet3D, self).__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels

        if reduction != 0.0:
            assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0.'

        assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4.'
        count = int((depth - 4) / 3)

        if bottleneck:
            count = count // 2

        num_layers = [count for _ in range(num_dense_block)]

        num_filters = 2 * growth_rate
        compression = 1.0 - reduction

        if subsample_initial_block:
            initial_kernel = (5, 5, 3)
            initial_strides = (2, 2, 1)
        else:
            initial_kernel = (3, 3, 1)
            initial_strides = (1, 1, 1)

        layers = []
        if subsample_initial_block:
            conv_layer = nn.Conv3d(1,
                                   num_filters,
                                   initial_kernel,
                                   stride=initial_strides,
                                   padding=(2, 2, 1),
                                   bias=False)
        else:
            conv_layer = nn.Conv3d(1,
                                   num_filters,
                                   initial_kernel,
                                   stride=initial_strides,
                                   padding=(1, 1, 0),
                                   bias=False)
        layers.append(("conv1", conv_layer))

        if subsample_initial_block:
            layers.append(("batch1", nn.BatchNorm3d(num_filters, eps=1.1e-5)))
            layers.append(("active1", nn.ReLU()))
            layers.append(("maxpool",
                           nn.MaxPool3d((2, 2, 2),
                                        stride=(2, 2, 2),
                                        padding=(0, 0, 1))))
        self.conv_layer = nn.Sequential(OrderedDict(layers))

        grid_height, grid_width, grid_channels = self.get_feature_dims()

        layers = []
        for block_idx in range(num_dense_block - 1):

            layers.append(
                Attention(grid_size=(grid_height, grid_width),
                          in_channels=grid_channels))

            layers.append(
                DenseBlock(num_layers[block_idx],
                           num_filters,
                           growth_rate,
                           bottleneck=bottleneck,
                           dropout=dropout))
            num_filters = num_filters + growth_rate * num_layers[block_idx]

            layers.append(
                Transition(num_filters, num_filters, compression=compression))
            num_filters = int(num_filters * compression)

            grid_height = int(grid_height / 2)
            grid_width = int(grid_width / 2)
            grid_channels = int(grid_channels / 2)

        layers.append(
            Attention(grid_size=(grid_height, grid_width),
                      in_channels=grid_channels))
        layers.append(
            DenseBlock(num_layers[block_idx],
                       num_filters,
                       growth_rate,
                       bottleneck=bottleneck,
                       dropout=dropout))
        num_filters = num_filters + growth_rate * num_layers[block_idx]

        self.layers = nn.ModuleList(layers)

        final_layers = []

        final_layers.append(nn.BatchNorm3d(num_filters, eps=1.1e-5))
        final_layers.append(nn.ReLU())
        final_layers.append(
            nn.AvgPool3d((grid_height, grid_width, grid_channels)))
        self.final_layers = nn.ModuleList(final_layers)

    def get_feature_dims(self):
        mock_eeg = torch.randn(2, self.grid_size[0], self.grid_size[1],
                               self.in_channels)
        mock_eeg = mock_eeg.unsqueeze(1)
        mock_eeg = self.conv_layer(mock_eeg)
        return mock_eeg.shape[2], mock_eeg.shape[3], mock_eeg.shape[4]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layer(x)

        for layer in self.layers:
            x = layer(x)

        for layer in self.final_layers:
            x = layer(x)

        x = x.view(x.shape[0], -1)

        return x


class DenseBlock(nn.Module):
    def __init__(self,
                 num_layers,
                 num_filters,
                 growth_rate,
                 bottleneck=False,
                 dropout=None):
        super(DenseBlock, self).__init__()

        layers = []

        for i in range(num_layers):
            convLayer = ConvBlock(num_filters, growth_rate, bottleneck, dropout)
            num_filters = num_filters + growth_rate
            layers.append(convLayer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            cb = layer(x)
            x = torch.cat([x, cb], dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 input_channel,
                 num_filters,
                 bottleneck=False,
                 dropout=None,
                 conv1x1=True):
        super(ConvBlock, self).__init__()

        layers = []

        layers.append(nn.BatchNorm3d(input_channel, eps=1.1e-5))
        layers.append(nn.ReLU())

        if bottleneck:
            inter_channel = num_filters * 4

            layers.append(
                nn.Conv3d(input_channel,
                          inter_channel, (1, 1, 1),
                          padding=0,
                          bias=False))
            layers.append(nn.BatchNorm3d(inter_channel, eps=1.1e-5))
            layers.append(nn.ReLU())

        layers.append(
            nn.Conv3d(inter_channel,
                      num_filters, (3, 3, 1),
                      padding=(1, 1, 0),
                      bias=False))

        if conv1x1:
            layers.append(
                nn.Conv3d(num_filters,
                          num_filters, (1, 1, 1),
                          padding=(0, 0, 0),
                          bias=False))

        layers.append(
            nn.Conv3d(num_filters,
                      num_filters, (1, 1, 3),
                      padding=(0, 0, 1),
                      bias=False))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, input_channel, num_filters, compression=1.0):
        super(Transition, self).__init__()

        layers = []

        layers.append(nn.BatchNorm3d(input_channel, eps=1.1e-5))
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv3d(input_channel,
                      int(num_filters * compression), (1, 1, 1),
                      padding=0,
                      bias=False))
        layers.append(nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, grid_size, in_channels):
        super(Attention, self).__init__()

        num_spatial = int(grid_size[0]) * int(grid_size[1])
        self.spatial_pool = nn.AvgPool3d(kernel_size=[1, 1, in_channels])
        self.spatail_dense = nn.Linear(num_spatial, num_spatial)

        self.temporal_pool = nn.AvgPool3d(
            kernel_size=[grid_size[0], grid_size[1], 1])
        self.temporal_dense = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        out = x

        x = torch.mean(x, dim=1)
        x = x.unsqueeze(1)

        num_spatial = x.shape[2] * x.shape[3]
        num_temporal = x.shape[-1]

        spatial = self.spatial_pool(x)
        spatial = spatial.view(-1, num_spatial)
        spatial = self.spatail_dense(spatial)
        spatial = F.sigmoid(spatial)
        spatial = spatial.view(x.shape[0], 1, x.shape[2], x.shape[3], 1)

        out = out * spatial

        temporal = self.temporal_pool(x)
        temporal = temporal.view(-1, num_temporal)
        temporal = self.temporal_dense(temporal)
        temporal = F.sigmoid(temporal)
        temporal = temporal.view(x.shape[0], 1, 1, 1, x.shape[-1])

        out = out * temporal

        return out