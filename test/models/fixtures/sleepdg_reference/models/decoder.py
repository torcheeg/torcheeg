import torch
import torch.nn as nn
from .transformer import TransformerEncoder


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        # self.mlp = nn.Sequential(
        #     nn.Linear(512, 512*2),
        #     nn.GELU(),
        #     nn.Linear(512*2, 512*5),
        #     nn.Dropout(params.dropout),
        # )
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(512, 256, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(256, 128, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(64, 2, kernel_size=49, stride=15, padding=17, bias=False),
        )
    def forward(self, x):
        bz = x.shape[0]
        # x = self.mlp(x)
        # print(x.shape)
        x = x.view(bz*20, 512, 1)
        x = self.upsample(x)
        # print(x.shape)
        return x.view(bz, 20, 2, 3000)
