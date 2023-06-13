import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, weight_norm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm:
            self.weight.data = torch.renorm(self.weight.data,
                                            p=2,
                                            dim=0,
                                            maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, weight_norm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm:
            self.weight.data = torch.renorm(self.weight.data,
                                            p=2,
                                            dim=0,
                                            maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


class VarLayer(nn.Module):
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


class StdLayer(nn.Module):
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim=self.dim, keepdim=True)


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(
            torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class MeanLayer(nn.Module):
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


class MaxLayer(nn.Module):
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma, ima = x.max(dim=self.dim, keepdim=True)
        return ma


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNet(nn.Module):
    r'''
    An Efficient Multi-view Convolutional Neural Network for Brain-Computer Interface. For more details, please refer to the following information.

    - Paper: Mane R, Chew E, Chua K, et al. FBCNet: A multi-view convolutional neural network for brain-computer interface[J]. arXiv preprint arXiv:2104.01233, 2021.
    - URL: https://arxiv.org/abs/2104.01233
    - Related Project: https://github.com/ravikiran-mane/FBCNet

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    chunk_size=512,
                    num_baseline=1,
                    baseline_chunk_size=512,
                    offline_transform=transforms.BandSignal(),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = FBCNet(num_classes=2,
                       num_electrodes=32,
                       chunk_size=512,
                       in_channels=4,
                       num_S=32)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`28`)
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`1000`)
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`9`)
        num_S (int): The number of spatial convolution block. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        temporal (str): The temporal layer used, with options including VarLayer, StdLayer, LogVarLayer, MeanLayer, and MaxLayer, used to compute statistics using different techniques in the temporal dimension. (default: :obj:`LogVarLayer`)
        stride_factor (int): The stride factor. (default: :obj:`4`)
        weight_norm (bool): Whether to use weight renormalization technique in Conv2dWithConstraint. (default: :obj:`True`)
    '''
    def __init__(self,
                 num_electrodes: int = 20,
                 chunk_size: int = 1000,
                 in_channels: int = 9,
                 num_S: int = 32,
                 num_classes: int = 2,
                 temporal: str = 'LogVarLayer',
                 stride_factor: int = 4,
                 weight_norm: bool = True):
        super(FBCNet, self).__init__()

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_S = num_S
        self.temporal = temporal
        self.stride_factor = stride_factor
        self.weight_norm = weight_norm

        assert chunk_size % stride_factor == 0, f'chunk_size should be divisible by stride_factor, chunk_size={chunk_size},stride_factor={stride_factor} does not meet the requirements.'

        self.scb = self.SCB(num_S,
                            num_electrodes,
                            self.in_channels,
                            weight_norm=weight_norm)

        if temporal == 'VarLayer':
            self.temporal_layer = VarLayer(dim=3)
        elif temporal == 'StdLayer':
            self.temporal_layer = StdLayer(dim=3)
        elif temporal == 'LogVarLayer':
            self.temporal_layer = LogVarLayer(dim=3)
        elif temporal == 'MeanLayer':
            self.temporal_layer = MeanLayer(dim=3)
        elif temporal == 'MaxLayer':
            self.temporal_layer = MaxLayer(dim=3)
        else:
            raise NotImplementedError

        self.last_layer = self.last_block(self.num_S * self.in_channels *
                                          self.stride_factor,
                                          num_classes,
                                          weight_norm=weight_norm)

    def SCB(self, num_S, num_electrodes, in_channels, weight_norm=True):
        return nn.Sequential(
            Conv2dWithConstraint(in_channels,
                                 num_S * in_channels, (num_electrodes, 1),
                                 groups=in_channels,
                                 max_norm=2,
                                 weight_norm=weight_norm,
                                 padding=0),
            nn.BatchNorm2d(num_S * in_channels), swish())

    def last_block(self, in_channels, out_channels, weight_norm=True):
        return nn.Sequential(
            LinearWithConstraint(in_channels,
                                 out_channels,
                                 max_norm=0.5,
                                 weight_norm=weight_norm), nn.LogSoftmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, in_channel, num_electrodes, chunk_size]`. Here, :obj:`n` corresponds to the batch size

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.scb(x)
        x = x.reshape([
            *x.shape[0:2], self.stride_factor,
            int(x.shape[3] / self.stride_factor)
        ])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        return x
