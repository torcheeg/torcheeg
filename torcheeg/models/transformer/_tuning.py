import math

import torch
from torch import nn
from torch.nn import functional as F


class _Adapter(nn.Module):
    def __init__(self, hid_channels, adapter_size, dropout):
        super().__init__()
        self.down_project = nn.Linear(hid_channels, adapter_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_project = nn.Linear(adapter_size, hid_channels)
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return x + self.up_project(self.dropout(self.activation(self.down_project(x))))


class _LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(linear.weight.new_empty(self.in_features, rank))
        self.lora_B = nn.Parameter(linear.weight.new_zeros(rank, self.out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.T, a=math.sqrt(5))
        self.train(linear.training)

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_A.T), self.lora_B.T)
        return F.linear(x, self.weight, self.bias) + update * self.scaling

