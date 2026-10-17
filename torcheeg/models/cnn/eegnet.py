from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _temporal_same_padding(kernel_size: int) -> Tuple[int, int, int, int]:
    r'''
    Zero padding along the time axis that keeps the output length equal to the
    input length for a stride-1 convolution, i.e. the :obj:`padding='SAME'`
    behavior of the reference TensorFlow implementation of EEGNet.

    :obj:`nn.Conv2d` can only pad symmetrically, so :obj:`padding=kernel_size // 2`
    adds :obj:`kernel_size` instead of :obj:`kernel_size - 1` samples whenever the
    kernel is even, which lengthens the time dimension by one. Splitting the
    padding by hand avoids that. For odd kernels the result is symmetric and
    matches the previous behavior, for even kernels the extra sample goes to the
    right, as TensorFlow does.

    Args:
        kernel_size (int): The kernel size along the time axis.

    Returns:
        Tuple[int, int, int, int]: The left, right, top and bottom padding to be
        passed to :obj:`nn.ZeroPad2d`.
    '''
    total_padding = kernel_size - 1
    left_padding = total_padding // 2
    return (left_padding, total_padding - left_padding, 0, 0)


class _Conv2dWithSamePadding(nn.Conv2d):
    r'''Preserve time length without adding a module to the checkpoint layout.'''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, _temporal_same_padding(self.kernel_size[1]))
        return super().forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0,
                                           maxnorm=self.max_norm))
        # Keep saved weights valid when several forwards share one backward.
        return self._conv_forward(x, self.weight.clone(), self.bias)


class EEGNet(nn.Module):
    r'''
    A compact convolutional neural network (EEGNet). For more details, please refer to the following information.

    - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    - URL: https://arxiv.org/abs/1611.08024
    - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.models import EEGNet
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor(),
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       F2=16,
                       D=2,
                       num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Temporal convolutions preserve the input time length, including for even
    kernels. Checkpoint parameter names remain compatible with earlier releases.
    Correcting the padding can change the final feature size for some chunk
    sizes (for example, 127 samples with the default kernels); such checkpoints
    require a new classification head. Even when tensor shapes match, even-kernel
    padding changes the model's numerical output.

    Maximum-norm constraints keep parameter storage and checkpoint names
    unchanged. For inference export, set the model to evaluation mode and
    perform :obj:`torch.export.export` within :obj:`torch.no_grad`. Validate
    the exported program on the target PyTorch version before deployment.

    .. code-block:: python

        model = EEGNet(num_electrodes=32, chunk_size=128).eval()
        example = torch.randn(2, 1, 32, 128)
        with torch.no_grad():
            program = torch.export.export(model, (example,), strict=True)
        torch.export.save(program, 'eegnet.pt2')

    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`16`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            _Conv2dWithSamePadding(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            _Conv2dWithSamePadding(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, 0),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation with shape
                :obj:`[batch_size, 1, num_electrodes, chunk_size]`.

        Returns:
            torch.Tensor: Classification logits with shape
                :obj:`[batch_size, num_classes]`.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x