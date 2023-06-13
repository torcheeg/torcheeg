import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ChannelAttention(nn.Module):
    def __init__(self, in_channelsnel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channelsnel,
                             in_channelsnel // ratio,
                             1,
                             bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channelsnel // ratio,
                             in_channelsnel,
                             1,
                             bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)
        avg_pool_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_pool_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = max_pool_out + avg_pool_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7.'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([avg_pool_out, max_pool_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, in_channelsnel, ratio=2, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.cha_att = ChannelAttention(in_channelsnel, ratio=ratio)
        self.spa_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.cha_att(x)
        out = out * self.spa_att(out)
        return out


class PowerLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super(PowerLayer, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size),
                                    stride=(1, stride))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class Aggregator():
    def __init__(self, region_list):
        self.region_list = region_list

    def forward(self, x):
        output = []
        for region_index in range(len(self.region_list)):
            region_x = x[:, self.region_list[region_index], :]
            aggr_region_x = torch.mean(region_x, dim=1)
            output.append(aggr_region_x)
        return torch.stack(output, dim=1)


class GraphConvolution(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(
                torch.zeros((1, 1, out_channels), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight, gain=1.414)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class LGGNet(nn.Module):
    r'''
    DLocal-Global-Graph Networks (LGGNet). For more details, please refer to the following information.

    - Paper: Ding Y, Robinson N, Zeng Q, et al. LGGNet: learning from Local-global-graph representations for brain-computer interface[J]. arXiv preprint arXiv:2105.02786, 2021.
    - URL: https://arxiv.org/abs/2105.02786
    - Related Project: https://github.com/yi-ding-cs/LGG

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              offline_transform=transforms.Compose([
                                  transforms.MeanStdNormalize(),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
        model = LGGNet(region_list=SEED_GENERAL_REGION_LIST, chunk_size=128, num_electrodes=32, hid_channels=32, num_classes=2)

    The current built-in :obj:`region_list` includs:

    - torcheeg.datasets.constants.emotion_recognition.amigos.AMIGOS_GENERAL_REGION_LIST
    - torcheeg.datasets.constants.emotion_recognition.amigos.AMIGOS_FRONTAL_REGION_LIST
    - torcheeg.datasets.constants.emotion_recognition.amigos.AMIGOS_HEMISPHERE_REGION_LIST
    - torcheeg.datasets.constants.emotion_recognition.deap.DEAP_GENERAL_REGION_LIST
    - ...
    - torcheeg.datasets.constants.emotion_recognition.dreamer.DREAMER_GENERAL_REGION_LIST
    - ...
    - torcheeg.datasets.constants.emotion_recognition.mahnob.MAHNOB_GENERAL_REGION_LIST
    - ...
    - torcheeg.datasets.constants.emotion_recognition.seed.SEED_GENERAL_REGION_LIST
    - ...

    Args:
        region_list (list): The local graph structure defined according to the 10-20 system, where the electrodes are divided into different brain regions.
        in_channels (int): The feature dimension of each electrode. (default: :obj:`1`)
        num_electrodes (int): The number of electrodes. (default: :obj:`32`)
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`128`)
        sampling_rate (int): The sampling rate of the EEG signals, i.e., :math:`f_s` in the paper. (default: :obj:`128`)
        num_T (int): The number of multi-scale 1D temporal kernels in the dynamic temporal layer, i.e., :math:`T` kernels in the paper. (default: :obj:`64`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.5`)
        pool_kernel_size (int): The kernel size of pooling layers in the temporal blocks (default: :obj:`16`)
        pool_stride (int): The stride of pooling layers in the temporal blocks (default: :obj:`4`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 region_list,
                 in_channels: int = 1,
                 num_electrodes: int = 32,
                 chunk_size: int = 128,
                 sampling_rate: int = 128,
                 num_T: int = 64,
                 hid_channels: int = 32,
                 dropout: float = 0.5,
                 pool_kernel_size: int = 16,
                 pool_stride: int = 4,
                 num_classes: int = 2):
        super(LGGNet, self).__init__()
        self.region_list = region_list
        self.inception_window = [0.5, 0.25, 0.125]

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.num_T = num_T
        self.hid_channels = hid_channels
        self.dropout = dropout
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes

        self.t_block1 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[0] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)
        self.t_block2 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[1] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)
        self.t_block3 = self.temporal_block(
            self.in_channels, self.num_T,
            (1, int(self.inception_window[2] * self.sampling_rate)),
            self.pool_kernel_size, self.pool_stride)

        self.bn_t1 = nn.BatchNorm2d(self.num_T)
        self.bn_t2 = nn.BatchNorm2d(self.num_T)

        self.cbam = CBAMBlock(num_electrodes)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(), nn.AvgPool2d((1, 2)))

        self.avg_pool = nn.AvgPool2d((1, 2))

        feature_dim = self.feature_dim
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(
            self.num_electrodes, feature_dim),
                                                requires_grad=True)
        self.local_filter_bias = nn.Parameter(torch.zeros(
            (1, self.num_electrodes, 1), dtype=torch.float32),
                                              requires_grad=True)

        self.aggregate = Aggregator(self.region_list)
        num_region = len(self.region_list)

        self.global_adj = nn.Parameter(torch.FloatTensor(
            num_region, num_region),
                                       requires_grad=True)

        self.bn_g1 = nn.BatchNorm1d(num_region)
        self.bn_g2 = nn.BatchNorm1d(num_region)

        self.gcn = GraphConvolution(feature_dim, hid_channels)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(num_region * hid_channels), num_classes))

        nn.init.xavier_uniform_(self.local_filter_weight)
        nn.init.xavier_uniform_(self.global_adj)

    def temporal_block(self, in_channels, out_channels, kernel_size,
                       pool_kernel_size, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=(1, 1)),
            PowerLayer(kernel_size=pool_kernel_size, stride=pool_stride))

    def forward(self, x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`chunk_size` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        t1 = self.t_block1(x)
        t2 = self.t_block2(x)
        t3 = self.t_block3(x)
        x = torch.cat((t1, t2, t3), dim=-1)

        x = self.bn_t1(x)

        x = x.permute(0, 2, 1, 3)
        x = self.cbam(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=2)
        x = self.local_filter(x)
        x = self.aggregate.forward(x)
        adj = self.get_adj(x)
        x = self.bn_g1(x)

        x = self.gcn(x, adj)
        x = self.bn_g2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    @property
    def feature_dim(self):
        mock_eeg = torch.randn(
            (1, self.in_channels, self.num_electrodes, self.chunk_size))

        t1 = self.t_block1(mock_eeg)
        t2 = self.t_block2(mock_eeg)
        t3 = self.t_block3(mock_eeg)
        mock_eeg = torch.cat((t1, t2, t3), dim=-1)

        mock_eeg = self.bn_t1(mock_eeg)
        mock_eeg = self.conv1x1(mock_eeg)
        mock_eeg = self.bn_t2(mock_eeg)
        mock_eeg = mock_eeg.permute(0, 2, 1, 3)
        mock_eeg = mock_eeg.flatten(start_dim=2)
        return mock_eeg.shape[-1]

    def local_filter(self, x):
        w = self.local_filter_weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        adj = torch.bmm(x, x.permute(0, 2, 1))
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(x.device)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj