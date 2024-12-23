import numpy as np
import torch
from torch import nn


def _crop_tensors_to_match(x1, x2, axis=-1):
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0)
        self.maxpool = nn.MaxPool1d(
            kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 with_skip_connection=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=2,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = _crop_tensors_to_match(
                x, residual, axis=-1)
            x = torch.cat([x, residual], axis=1)
        x = self.block_postskip(x)
        return x


class USleep(nn.Module):
    r'''
    A publicly available, ready-to-use deep-learning-based system for automated sleep staging. For more details, please refer to the following information.

    - Paper: Perslev M, Darkner S, Kempfner L, et al. U-Sleep: resilient high-frequency sleep staging[J]. NPJ digital medicine, 2021, 4(1): 72.
    - URL: https://www.nature.com/articles/s41746-021-00440-5
    - Related Project: https://github.com/perslev/U-Time
    - Related Project: https://github.com/braindecode/braindecode/blob/master/braindecode/models/usleep.py

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import USleep

        model = USleep(num_electrodes=1,
                      patch_size=100, 
                      num_patchs=30,
                      num_classes=5)

        # batch_size, num_electrodes, num_patchs * patch_size
        x = torch.randn(32, 1, 3000)
        model(x)

    Args:
        num_electrodes (int): The number of EEG channels. (default: :obj:`1`)
        patch_size (int): The size of each patch that the signal will be divided into. (default: :obj:`100`)
        depth (int): The depth of the U-Net architecture, determining the number of encoder and decoder blocks. (default: :obj:`12`)
        num_filters (int): The initial number of filters, which will be increased through the network. (default: :obj:`5`)
        complexity_factor (float): Factor controlling the growth of filter numbers across layers. (default: :obj:`1.67`)
        with_skip_connection (bool): Whether to use skip connections between encoder and decoder. (default: :obj:`True`)
        num_classes (int): The number of sleep stages to classify. (default: :obj:`5`)
        num_patchs (int): The number of patches to divide the input signal into. (default: :obj:`30`)
        filter_size (int): The size of convolutional filters, must be odd. (default: :obj:`9`)
    '''

    def __init__(self,
                 num_electrodes: int = 1,
                 patch_size: int = 100,
                 depth: int = 12,
                 num_filters: int = 5,
                 complexity_factor: float = 1.67,
                 with_skip_connection: bool = True,
                 num_classes: int = 5,
                 num_patchs: int = 30,
                 filter_size: int = 9,
                 ):
        super().__init__()

        self.num_electrodes = num_electrodes
        max_pool_size = 2
        if filter_size % 2 == 0:
            raise ValueError(
                'filter_size must be an odd number to accomodate the '
                'upsampling step in the decoder blocks.')

        input_size = np.ceil(num_patchs * patch_size).astype(int)

        channels = [num_electrodes]
        n_filters = num_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        encoder = list()
        for idx in range(depth):
            encoder += [
                _EncoderBlock(in_channels=channels[idx],
                              out_channels=channels[idx + 1],
                              kernel_size=filter_size,
                              downsample=max_pool_size)
            ]
        self.encoder = nn.Sequential(*encoder)

        self.bottom = nn.Sequential(
            nn.Conv1d(in_channels=channels[-2],
                      out_channels=channels[-1],
                      kernel_size=filter_size,
                      padding=(filter_size - 1) // 2),  # preserves dimension
            nn.ELU(),
            nn.BatchNorm1d(num_features=channels[-1]),
        )

        decoder = list()
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                _DecoderBlock(in_channels=channels_reverse[idx],
                              out_channels=channels_reverse[idx + 1],
                              kernel_size=filter_size,
                              upsample=max_pool_size,
                              with_skip_connection=with_skip_connection)
            ]
        self.decoder = nn.Sequential(*decoder)

        self.emb_dim = channels[1]
        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
            nn.AvgPool1d(input_size),
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_classes,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 2, 1, 3)
            x = x.flatten(start_dim=2)

        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        x = self.bottom(x)

        residuals = residuals[::-1]
        for up, res in zip(self.decoder, residuals):
            x = up(x, res)

        y_pred = self.clf(x)

        if y_pred.shape[-1] == 1:
            y_pred = y_pred[:, :, 0]

        return y_pred
