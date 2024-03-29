import torch.nn.functional as F
from typing import Tuple, Optional
import torch
import torch.nn as nn
from .fbcnet import VarLayer, MaxLayer, StdLayer, LogVarLayer, LinearWithConstraint, MeanLayer, swish, Conv2dWithConstraint


## CONV_SAME_PADDING
def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _same_pad_arg(input_size, kernel_size, stride, dilation, **_):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def conv2d_same(x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                stride: Tuple[int, int] = (1, 1),
                padding: Tuple[int, int] = (0, 0),
                dilation: Tuple[int, int] = (1, 1),
                groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x,
              [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class SamePadConv2d(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(SamePadConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride, 0,
                             dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding,
                           self.dilation, self.groups)


## MIX_CONV
def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return SamePadConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size, tuple):
            padding = (0, padding)
        return nn.Conv2d(in_chs,
                         out_chs,
                         kernel_size,
                         padding=padding,
                         **kwargs)


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            dynamic = True
            padding = 0

        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            dynamic = True
            padding = 0
    else:
        dynamic = True
        padding = 0
    return padding, dynamic


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 dilation=1,
                 depthwise=False,
                 **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size,
                                                list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch,
                  out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(in_ch,
                                  out_ch,
                                  k,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=conv_groups,
                                  **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


class FBMSNet(nn.Module):
    r'''
        FBMSNet, a novel multiscale temporal convolutional neural network for MI decoding tasks, employs Mixed Conv to extract multiscale temporal features which  enhance the intra-class compactness and improve the inter-class separability with the joint supervision of the center loss andcenter loss.

        - Paper: FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery Decoding
        - URL: https://ieeexplore.ieee.org/document/9837422
        - Related Project: https://github.com/Want2Vanish/FBMSNet

        Below is a example to explain how to use this model. Firstly we should transform eeg signal to several nonoverlapping frequency bands by :obj:`torcheeg.transforms.BandSignal` 

        .. code-block:: python

            from torcheeg.datasets import BCICIV2aDataset
            from torcheeg import transforms
            from torcheeg.models import FBMSNet
            from torch.utils.data import DataLoader

            freq_range_per_band = {
                'sub band1': [4, 8],
                'sub band2': [8, 12],
                'sub band3': [12, 16],
                'sub band4': [16, 20],
                'sub band5': [20, 24],
                'sub band6': [24, 28],
                'sub band7': [28, 32],
                'sub band8': [32, 36],
                'sub band9': [36, 40]
            }
            dataset = BCICIV2aDataset(root_path='./BCICIV_2a_mat',
                                      chunk_size=512,
                                      offline_transform=transforms.BandSignal(band_dict=freq_range_per_band,
                                                                              sampling_rate=250),
                                      online_transform=transforms.ToTensor(),
                                      label_transform=transforms.Compose(
                                          [transforms.Select('label'),
                                          transforms.Lambda(lambda x: x - 1)]))

            model = FBMSNet(num_classes=4, num_electrodes=22, chunk_size=512, in_channels=9)

            x, y = next(iter(DataLoader(dataset, batch_size=64)))
            model(x)
            
        Args:
            num_electrodes (int): The number of electrodes. 
            chunk_size (int): Number of data points included in each EEG chunk. 
            in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`9`)
            num_classes (int): The number of classes to predict. (default: :obj:`4`)
            stride_factor (int): The stride factor. Please make sure the chunk_size parameter is a  multiple of stride_factor parameter in order to init model successfully. (default: :obj:`4`)
            temporal (str): The temporal layer used, with options including VarLayer, StdLayer, LogVarLayer, MeanLayer, and MaxLayer, used to compute statistics using different techniques in the temporal dimension. (default: :obj:`LogVarLayer`)
            num_feature (int): The number of Mixed Conv output channels which can stand for various kinds of feature. (default: :obj:`36`)
            dilatability (int): The expansion multiple of the channels after the input bands pass through spatial convolutional blocks. (default: :obj:`8`
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

            self.center_dim = self.feature_dim(in_channels, num_electrodes,
                                               chunk_size)[-1]

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

    def forward(self, x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, in_channel, num_electrodes, chunk_size ]`. Here, :obj:`n` corresponds to the batch size

        Returns:
            torch.Tensor[size of batch,number of classes]: The predicted probability that the samples belong to the classes.
        '''
        x = self.mixConv2d(x)
        x = self.scb(x)
        x = x.reshape([
            *x.shape[0:2], self.stride_factor,
            int(x.shape[3] / self.stride_factor)
        ])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc(x)

    def feature_dim(self, in_channels, num_electrodes, chunk_size):
        data = torch.ones((1, in_channels, num_electrodes, chunk_size))
        x = self.mixConv2d(data)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.stride_factor, -1])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()
