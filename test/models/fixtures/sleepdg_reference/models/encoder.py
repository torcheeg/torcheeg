import torch
import torch.nn as nn
from .transformer import TransformerEncoder


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.epoch_encoder = EpochEncoder(self.params)
        self.seq_encoder = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )
        self.fc_mu = nn.Linear(512, 512)
        # self.fc_log_var  = nn.Linear(512, 512)

    def forward(self, x):
        bz = x.shape[0]

        x = x.view(bz*20, 2, 3000)
        x = self.epoch_encoder(x)
        x_epoch = x.view(bz, 20, -1)

        x_seq = self.seq_encoder(x_epoch)
        mu = self.fc_mu(x_seq)
        # log_var = self.fc_log_var(x_seq)
        return mu


class EpochEncoder(nn.Module):
    def __init__(self, params):
        super(EpochEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(params.dropout),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        # self.layer_norm = LayerNorm(512)
    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        # print(x.shape)
        x = self.avg(x).squeeze()
        return x
