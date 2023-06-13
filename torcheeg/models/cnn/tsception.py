import torch
import torch.nn as nn


class TSCeption(nn.Module):
    r'''
    Continuous Convolutional Neural Network (CCNN). For more details, please refer to the following information.

    - Paper: Ding Y, Robinson N, Zhang S, et al. Tsception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition[J]. arXiv preprint arXiv:2104.02935, 2021.
    - URL: https://arxiv.org/abs/2104.02935
    - Related Project: https://github.com/yi-ding-cs/TSception

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    chunk_size=512,
                    num_baseline=1,
                    baseline_chunk_size=512,
                    offline_transform=transforms.Compose([
                        PickElectrode(PickElectrode.to_index_list(
                        ['FP1', 'AF3', 'F3', 'F7',
                        'FC5', 'FC1', 'C3', 'T7',
                        'CP5', 'CP1', 'P3', 'P7',
                        'PO3','O1', 'FP2', 'AF4',
                        'F4', 'F8', 'FC6', 'FC2',
                        'C4', 'T8', 'CP6', 'CP2',
                        'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
                        transforms.To2d()
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = TSCeption(num_classes=2,
                          num_electrodes=28,
                          sampling_rate=128,
                          num_T=15,
                          num_S=15,
                          hid_channels=32,
                          dropout=0.5)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`28`)
        num_T (int): The number of multi-scale 1D temporal kernels in the dynamic temporal layer, i.e., :math:`T` kernels in the paper. (default: :obj:`15`)
        num_S (int): The number of multi-scale 1D spatial kernels in the asymmetric spatial layer. (default: :obj:`15`)
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`1`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        sampling_rate (int): The sampling rate of the EEG signals, i.e., :math:`f_s` in the paper. (default: :obj:`128`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.5`)
    '''
    def __init__(self,
                 num_electrodes: int = 28,
                 num_T: int = 15,
                 num_S: int = 15,
                 in_channels: int = 1,
                 hid_channels: int = 32,
                 num_classes: int = 2,
                 sampling_rate: int = 128,
                 dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(TSCeption, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))

    def conv_block(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 28, 512]`. Here, :obj:`n` corresponds to the batch size, :obj:`1` corresponds to number of channels for convolution, :obj:`28` corresponds to :obj:`num_electrodes`, and :obj:`512` corresponds to the input dimension for each electrode.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
