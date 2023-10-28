import torch
import torch.nn as nn
from ..any.custom_layer import MixedConv2d, Conv2dWithConstraint, LinearWithConstraint


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
                                 weight_norm=weight_norm),
            nn.LogSoftmax(dim=1))

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

    def feature_dim(self):
        return self.num_S * self.in_channels * self.stride_factor


class FBMSNet(nn.Module):
    r'''
        FBMSNet, a novel multiscale temporal convolutional neural network for MI decoding tasks, employs Mixed Conv to extract multiscale temporal features which  enhance the intra-class compactness and improve the inter-class separability with the joint supervision of the center loss andcenter loss.

        - Paper: FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery Decoding
        - URL: https://ieeexplore.ieee.org/document/9837422
        - Related Project: https://github.com/Want2Vanish/FBMSNet

        Below is a example to explain how to use this model. Firstly we should transform eeg signal to several nonoverlapping frequency bands by :obj:`torcheeg.transforms.BandSignal` 

        .. code-block:: python

            # Define 9 nonoverlapping frequency bands, each with a 4 Hz bandwidth and spanning from 4 to 40 Hz.
            Freq_range_per_band = {'sub band1': [4, 8],
                                'sub band2': [8, 12],
                                'sub band3': [12, 16],
                                'sub band4': [16, 20],
                                'sub band5': [20, 24],
                                'sub band6': [24, 28],
                                'sub band7': [28, 32],
                                'sub band8': [32, 36],
                                'sub band9': [36, 40]}
            dataset =BCICIV2aDataset(io_path=f'./tmp_out/bciciv2a/band_9_filters',
                                    root_path='./BCICIV_2a_mat',
                                    chunk_size=512,
                                    offline_transform=transforms.BandSignal(band_dict=Freq_range_per_band,
                                                                            sampling_rate=250),
                                    online_transform=transforms.ToTensor(),
                                    label_transform=transforms.Compose([
                                    transforms.Select('label'),
                                    transforms.Lambda(lambda x:x-1),
                        ]))
            data = Dataloader(dataset)

            model = FBMSNet( num_classes=4,
                            num_electrodes=22,
                            chunk_size=512,
                            in_channels=9 )

        There are two ways to use the model. The first one, the effective way to get the prediction result but it don't output the decoded feature. 

        .. code-block:: python

            x,y = next(iter(data))
            pred = model(x)

        To obtain the decoded feature, use :obj:`decoder` method. If we want to obtain prediction results based on the encoded features, use :obj:`predict_by_feature` method.

        .. code-block:: python

            x,y = next(iter(data))
            feature = model.decoder(x)
            pred = model.predict_by_feature(feature)

        Args:
            num_electrodes (int): The number of electrodes. 
            chunk_size (int): Number of data points included in each EEG chunk. 
            in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`9`)
            num_classes (int): The number of classes to predict. (default: :obj:`4`)
            stride_factor (int): The stride factor. Please make sure the chunk_size parameter is a  multiple of stride_factor parameter in order to init model successfully. (default: :obj:`4`)
            temporal (str): The temporal layer used, with options including VarLayer, StdLayer, LogVarLayer, MeanLayer, and MaxLayer, used to compute statistics using different techniques in the temporal dimension. (default: :obj:`LogVarLayer`)
            num_feature (int): The number of Mixed Conv output channels which can stand for various kinds of feature. (default: :obj:`36`)
            dilatability (int): The expansion multiple of the channels after the input bands pass through spatial convolutional blocks. (default: :obj:`8`

        .. automethod:: decoder
        .. automethod:: predict_by_feature
    '''

    def __init__(self,
                 in_channels: int,
                 num_electrodes: int,
                 chunk_size: int,
                 num_classes: int = 4,
                 stride_factor: int = 4,
                 temporal: str = 'LogVarLayer',
                 num_feature: int = 36,
                 dilatability: int = 8):

        super(FBMSNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.stride_factor = stride_factor
        

        try:
            self.mixConv2d = nn.Sequential(
                MixedConv2d(in_channels=in_channels,
                            out_channels=num_feature,
                            kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
                            stride=1,
                            padding='',
                            dilation=1,
                            depthwise=False),
                nn.BatchNorm2d(num_feature),
            )
            self.scb = self.SCB(in_chan=num_feature,
                                out_chan=num_feature * dilatability,
                                num_electrodes=int(num_electrodes))

            # Formulate the temporal agreegator
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
            
            self.center_dim = self.feature_dim(
                in_channels, num_electrodes, chunk_size)[-1]

            self.fc = self.LastBlock(self.center_dim, num_classes)
        except:
            raise Exception(
                "Model init failed: The Chunksize must be a  multiple of stride_factor.Please modify values of stride_factor or chunk_size."
            )

    def SCB(self,
            in_chan,
            out_chan,
            num_electrodes,
            weight_norm=True,
            *args,
            **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(in_chan,
                                 out_chan, (num_electrodes, 1),
                                 groups=in_chan,
                                 max_norm=2,
                                 weight_norm=weight_norm,
                                 padding=0), nn.BatchNorm2d(out_chan), swish())

    def LastBlock(self, inF, outF, weight_norm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF,
                                 outF,
                                 max_norm=0.5,
                                 weight_norm=weight_norm,
                                 *args,
                                 **kwargs), nn.LogSoftmax(dim=1))

    def decoder(self, x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, in_channel, num_electrodes, chunk_size ]`. Here, :obj:`n` corresponds to the batch size, :obj:`in_channels` corresponds to the number of sub bands.

        Returns:
           torch.Tensor[size of batch, length of deep feature code]: The extracted deep features.
        '''
        x = self.mixConv2d(x)
        x = self.scb(x)
        x = x.reshape([
            *x.shape[0:2], self.stride_factor,
            int(x.shape[3] / self.stride_factor)
        ])
        x = self.temporal_layer(x)
        return torch.flatten(x, start_dim=1)

    def predict_by_feature(self, feature):
        r'''
        With feature which is ouput by decoder inputed,the predict_by_feature ouput the predicted probability that the samples belong to the classes. 

        Args:
            feature (torch.Tensor): The extracted deep features. The ideal input shape is :obj:`[batch size,1152]`where feature dim is fixed as :obj:`1152`.
        Returns:
           torch.Tensor[size of batch, num_classes]: The predicted probability that the samples belong to the classes.
        '''
        return self.fc(feature)

    def forward(self, x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, in_channel, num_electrodes, chunk_size ]`. Here, :obj:`n` corresponds to the batch size

        Returns:
            torch.Tensor[size of batch,number of classes]: The predicted probability that the samples belong to the classes.
        '''
        f = self.decoder(x)
        return self.predict_by_feature(f)

    def feature_dim(self, in_channels, num_electrodes, chunk_size):
        data = torch.ones((1, in_channels, num_electrodes, chunk_size))
        x = self.mixConv2d(data)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.stride_factor, -1])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()
