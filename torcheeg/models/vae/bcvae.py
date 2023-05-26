from typing import Tuple

import torch
import torch.nn as nn


class BCEncoder(nn.Module):
    r'''
    TorchEEG provides an EEG feature encoder based on CNN architecture and CVAE for generating EEG grid representations of different frequency bands based on a given class label. In particular, the expected labels are additionally provided to guide the encoder to derive the mean and standard deviation vectors of the given expected labels and input data.

    - Related Project: https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py

    .. code-block:: python

        encoder = BCEncoder(in_channels=4, num_classes=3)
        y = torch.randint(low=0, high=3, size=(1,))
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg, y)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        hid_channels (int): The number of hidden nodes in the first convolutional layer, which is also used as the dimension of output mu and logvar. (default: :obj:`32`)
        num_classes (int): The number of classes. (default: :obj:`2`)
    '''
    def __init__(self,
                 in_channels: int = 4,
                 grid_size: Tuple[int, int] = (9, 9),
                 hid_channels: int = 64,
                 num_classes: int = 3):
        super(BCEncoder, self).__init__()

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.label_embeding = nn.Embedding(
            num_classes, in_channels * grid_size[0] * grid_size[1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2,
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

        feature_dim = self.feature_dim()
        self.fc_mu = nn.Linear(feature_dim, self.hid_channels)
        self.fc_var = nn.Linear(feature_dim, self.hid_channels)

    def feature_dim(self):
        with torch.no_grad():
            mock_y = torch.randint(low=0, high=self.num_classes, size=(1, ))
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            label_emb = self.label_embeding(mock_y)
            label_emb = label_emb.reshape(mock_eeg.shape)
            mock_eeg = torch.cat([mock_eeg, label_emb], dim=1)

            mock_eeg = self.conv1(mock_eeg)
            mock_eeg = self.conv2(mock_eeg)
            mock_eeg = self.conv3(mock_eeg)
            mock_eeg = self.conv4(mock_eeg)

        return mock_eeg.flatten(start_dim=1).shape[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
            y (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size.

        Returns:
            tuple[2,]: The mean and standard deviation vectors obtained by encoder. The shapes of the feature vectors are all :obj:`[n, 64]`. Here, :obj:`n` corresponds to the batch size, and :obj:`64` corresponds to :obj:`hid_channels`.
        '''
        label_emb = self.label_embeding(y)
        label_emb = label_emb.reshape(x.shape)
        x = torch.cat([x, label_emb], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var


class BCDecoder(nn.Module):
    r'''
    TorchEEG provides an EEG feature decoder based on CNN architecture and CVAE for generating EEG grid representations of different frequency bands based on a given class label. In particular, the expected labels are additionally provided to guide the decoder to reconstruct samples of the specified class.

    - Related Project: https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py

    .. code-block:: python

        encoder = BCEncoder(in_channels=4, num_classes=3)
        decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=3)
        y = torch.randint(low=0, high=3, size=(1,))
        mock_eeg = torch.randn(1, 4, 9, 9)
        mu, logvar = encoder(mock_eeg, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        fake_X = decoder(z, y)

    Args:
        in_channels (int): The input feature dimension (of noise vectors). (default: :obj:`64`)
        out_channels (int): The generated feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
    '''
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 4,
                 grid_size: Tuple[int, int] = (9, 9),
                 num_classes: int = 3):
        super(BCDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.num_classes = num_classes

        self.label_embeding = nn.Embedding(num_classes, in_channels)
        self.deproj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4 * 3 * 3), nn.LeakyReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4,
                               in_channels * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
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
                               bias=True), nn.BatchNorm2d(in_channels),
            nn.LeakyReLU())
        self.deconv4 = nn.ConvTranspose2d(in_channels,
                                          out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): Given the mean and standard deviation vectors, the feature vector :obj:`z` obtained using the reparameterization technique. The shapes of the feature vector should be :obj:`[n, 64]`. Here, :obj:`n` corresponds to the batch size, and :obj:`64` corresponds to :obj:`in_channels`.
            y (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size.

        Returns:
            torch.Tensor[n, 4, 9, 9]: the decoded results, which should have the same shape as the input noise, i.e., :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
        '''
        label_emb = self.label_embeding(y)
        x = torch.cat([x, label_emb], dim=-1)
        x = self.deproj(x)
        x = x.view(-1, self.in_channels * 4, 3, 3)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x