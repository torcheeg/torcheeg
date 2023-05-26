import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels, grid_size):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.multiatt = nn.MultiheadAttention(in_channels, 4)
        self.layernorm = nn.LayerNorm([in_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels,
                   self.grid_size[0] * self.grid_size[1]).swapaxes(1, 2)
        x_layernorm = self.layernorm(x)

        x_layernorm = x_layernorm.transpose(0, 1)
        attention_value, _ = self.multiatt(x_layernorm, x_layernorm,
                                           x_layernorm)
        attention_value = attention_value.transpose(0, 1)

        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.in_channels,
                                                   self.grid_size[0],
                                                   self.grid_size[1])


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_channels, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None,
                                None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample_size,
                 hid_channels=256):
        super(Up, self).__init__()

        self.up = nn.Upsample(size=upsample_size,
                              mode="bilinear",
                              align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_channels, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None,
                                None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class BCUNet(nn.Module):
    r'''
    The diffusion model consists of two processes, the forward process, and the backward process. The forward process is to gradually add Gaussian noise to an image until it becomes random noise, while the backward process is the de-noising process. We train an attention-based UNet network at the backward process to start with random noise and gradually de-noise it until an image is generated and use the UNet to generate a simulated image from random noises. In particular, in conditional UNet, additional label information is provided to guide the noise reduction results during the noise reduction process.
    
    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.

    - Paper: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.
    - URL: https://arxiv.org/abs/2006.11239
    - Related Project: https://github.com/dome272/Diffusion-Models-pytorch

    Below is a recommended suite for use in EEG generation:

    .. code-block:: python

        .. code-block:: python

        noise = torch.randn(1, 4, 9, 9)
        t = torch.randint(low=1, high=1000, size=(1, ))
        y = torch.randint(low=0, high=2, size=(1, ))
        unet = BCUNet(num_classes=2)
        fake_X = unet(noise, t, y)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        hid_channels (int): The basic hidden channels in the network blocks. (default: :obj:`64`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`) 
        beta_timesteps (int): The variance schedule controlling step sizes. (default: :obj:`256`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 in_channels=4,
                 hid_channels=64,
                 grid_size=(9, 9),
                 beta_timesteps=256,
                 num_classes=2):
        super(BCUNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.beta_timesteps = beta_timesteps
        self.num_classes = num_classes

        self.label_embeding = nn.Embedding(num_classes, beta_timesteps)

        self.inc = DoubleConv(in_channels, hid_channels)
        self.down1 = Down(hid_channels, hid_channels * 2)

        att1_grid_size = (grid_size[0] // 2, grid_size[1] // 2)
        self.att1 = SelfAttention(hid_channels * 2, att1_grid_size)
        self.down2 = Down(hid_channels * 2, hid_channels * 4)

        att2_grid_size = (att1_grid_size[0] // 2, att1_grid_size[1] // 2)
        self.att2 = SelfAttention(hid_channels * 4, att2_grid_size)
        self.down3 = Down(hid_channels * 4, hid_channels * 4)

        att3_grid_size = (att2_grid_size[0] // 2, att2_grid_size[1] // 2)
        self.att3 = SelfAttention(hid_channels * 4, att3_grid_size)

        self.bot1 = DoubleConv(hid_channels * 4, hid_channels * 8)
        self.bot2 = DoubleConv(hid_channels * 8, hid_channels * 8)
        self.bot3 = DoubleConv(hid_channels * 8, hid_channels * 4)

        self.up1 = Up(hid_channels * 8, hid_channels * 2, att2_grid_size)
        self.att4 = SelfAttention(hid_channels * 2, att2_grid_size)

        self.up2 = Up(hid_channels * 4, hid_channels, att1_grid_size)
        self.att5 = SelfAttention(hid_channels, att1_grid_size)

        self.up3 = Up(hid_channels * 2, hid_channels, grid_size)
        self.att6 = SelfAttention(hid_channels, grid_size)

        self.outc = nn.Conv2d(hid_channels, in_channels, kernel_size=1)

    def position_encoding(self, t, channels):
        inv_freq = 1.0 / (10000**(
            torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        r'''
        Args:
            x (torch.Tensor): The random noise to be denoised, which should have the same shape as the simulated EEG expected to be generated, i.e., :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
            t (torch.Tensor): The randomly sampled time steps (int) for denoising a batch of samples. The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size.
            y (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size.

        Returns:
            torch.Tensor[n, 4, 9, 9]: the denoised results, which should have the same shape as the input noise, i.e., :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
        '''
        t = t.unsqueeze(-1).float()
        t = self.position_encoding(t, self.beta_timesteps)
        t += self.label_embeding(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)
        x4 = self.down3(x3, t)
        x4 = self.att3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.att4(x)
        x = self.up2(x, x2, t)
        x = self.att5(x)
        x = self.up3(x, x1, t)
        x = self.att6(x)
        output = self.outc(x)
        return output
