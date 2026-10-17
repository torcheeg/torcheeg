from numbers import Integral
from typing import Tuple

import torch
import torch.nn as nn


def _validate_grid_size(grid_size):
    try:
        valid = len(grid_size) == 2 and all(
            isinstance(size, Integral) and not isinstance(size, bool) and size > 0
            for size in grid_size)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError('grid_size must contain two positive integers.')


class BEncoder(nn.Module):
    r'''
    The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the encoder part.

    .. code-block:: python

        import torch

        from torcheeg.models import BEncoder
        
        encoder = BEncoder(in_channels=4)
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        hid_channels (int): The number of hidden nodes in the first convolutional layer, which is also used as the dimension of output mu and log-variance. (default: :obj:`64`)
    '''
    def __init__(self,
                 in_channels: int = 4,
                 grid_size: Tuple[int, int] = (9, 9),
                 hid_channels: int = 64):
        super(BEncoder, self).__init__()
        _validate_grid_size(grid_size)

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.hid_channels = hid_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      hid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(hid_channels), nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hid_channels,
                      hid_channels * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True), nn.BatchNorm2d(hid_channels * 2),
            nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid_channels * 2,
                      hid_channels * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(hid_channels * 2),
            nn.LeakyReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(hid_channels * 2,
                      hid_channels * 4,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True), nn.BatchNorm2d(hid_channels * 4),
            nn.LeakyReLU())

        feature_dim = self.feature_dim
        self.fc_mu = nn.Linear(feature_dim, self.hid_channels)
        self.fc_var = nn.Linear(feature_dim, self.hid_channels)

    @property
    def feature_dim(self):
        # Each stride-two convolution rounds the spatial dimension upward.
        height, width = ((size + 3) // 4 for size in self.grid_size)
        return self.hid_channels * 4 * height * width

    def forward(self, x: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            tuple[2,]: The mean and log-variance vectors obtained by encoder. Both have shape :obj:`[n, hid_channels]`.
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var


class BDecoder(nn.Module):
    r'''
    The variational autoencoder consists of two parts, an encoder, and a decoder. The encoder compresses the input into the latent space. The decoder receives as input the information sampled from the latent space and produces it as similar as possible to ground truth. The latent vector should approach the gaussian distribution supervised by KL divergence based on the variation trick. This class implement the decoder part.

    .. code-block:: python

        encoder = BEncoder(in_channels=4)
        decoder = BDecoder(in_channels=64, out_channels=4)
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z)

    The output spatial shape equals :obj:`grid_size`, including odd and even
    rectangular grids. Set the same grid size on the encoder and decoder.
    Existing default :obj:`(9, 9)` checkpoint parameter shapes are unchanged.
    Older decoders ignored :obj:`grid_size` and always returned a 9 by 9 grid.
    Load those decoder checkpoints with :obj:`grid_size=(9, 9)` to preserve
    their original architecture and output.

    Args:
        in_channels (int): The input feature dimension (of noise vectors). (default: :obj:`64`)
        out_channels (int): The generated feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
    '''
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 4,
                 grid_size: Tuple[int, int] = (9, 9)):
        super(BDecoder, self).__init__()
        _validate_grid_size(grid_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        height, width = grid_size
        self._projection_size = ((height + 3) // 4, (width + 3) // 4)
        intermediate_size = ((height + 1) // 2, (width + 1) // 2)

        self.deproj = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4 * self._projection_size[0] * self._projection_size[1]), nn.LeakyReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4,
                               in_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=tuple((size - 1) % 2 for size in intermediate_size),
                               bias=True), nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2,
                               in_channels * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True), nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU())
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2,
                               in_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=tuple((size - 1) % 2 for size in grid_size),
                               bias=True), nn.BatchNorm2d(in_channels),
            nn.LeakyReLU())
        self.deconv4 = nn.ConvTranspose2d(in_channels,
                                          out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

    def forward(self, x: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): The latent vector obtained from the mean and log-variance using reparameterization. Its shape is :obj:`[n, in_channels]`.

        Returns:
            torch.Tensor: The reconstruction with shape :obj:`[n, out_channels, grid_size[0], grid_size[1]]`.
        '''
        x = self.deproj(x)
        x = x.view(-1, self.in_channels * 4, *self._projection_size)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
