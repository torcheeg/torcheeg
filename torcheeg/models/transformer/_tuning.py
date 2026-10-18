from torch import nn


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
