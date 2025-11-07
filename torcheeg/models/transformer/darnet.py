import math
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Positional embedding layer for sequence data.

    Args:
        d_model (int): The dimension of the model.
        max_len (int): Maximum sequence length. (default: :obj:`5000`)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Token embedding layer with spatiotemporal construction.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        d_model (int): The dimension of the model.
    """

    def __init__(self, num_electrodes: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, d_model * 4, kernel_size=(1, 8), padding='same'),
            nn.BatchNorm2d(d_model * 4),
            nn.GELU()
        )

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model,
                      kernel_size=(num_electrodes, 1), padding='valid'),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)
        x = x.permute(0, 2, 1)
        x = x + self.position_embedding(x)
        return x


class Attention(nn.Module):
    """Multi-head attention mechanism.

    Args:
        emb_size (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len,
                                self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class Refine(nn.Module):
    """Refinement layer with 1D convolution and pooling.

    Args:
        emb_size (int): The embedding dimension.
    """

    def __init__(self, emb_size: int):
        super(Refine, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=3,
            padding=padding
        )
        self.norm = nn.BatchNorm1d(emb_size)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class AttnRefine(nn.Module):
    """Attention refinement block combining attention and convolutional refinement.

    Args:
        emb_size (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate for attention.
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = Attention(emb_size, num_heads, dropout)
        self.conv_layer = Refine(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_size, 4)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> tuple:
        x_src = self.attention(x)
        x_src = self.conv_layer(x_src)
        gap = self.gap(x_src.permute(0, 2, 1))
        out = self.out(self.flatten(gap))
        return x_src, out


class DARNet(nn.Module):
    r'''
    The DARNet model is based on the paper "DARNet: Dual Attention Refinement Network with Spatiotemporal Construction for Auditory Attention Detection". For more details, please refer to the following information.

    - Paper: Yan S, Fan C, Zhang H, et al. Darnet: Dual attention refinement network with spatiotemporal construction for auditory attention detection[J]. Advances in Neural Information Processing Systems, 2024, 37: 31688-31707.
    - URL: https://openreview.net/forum?id=jWGGEDYORs&noteId=0A27gTqMH0
    - Related Project: https://github.com/fchest/DARNet.git

    Below is a recommended suite for use in auditory attention detection tasks:

    .. code-block:: python

        from torcheeg.models import DARNet
        from torcheeg.datasets import DTUDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        dataset = DTUDataset(root_path='./DATA_preproc',
                              offline_transform=transforms.Compose([
                                  transforms.MinMaxNormalize(axis=-1),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('attended_speaker'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]))

        model = DARNet(num_electrodes=64,
                       chunk_size=64,
                       d_model=16,
                       num_heads=8,
                       attn_dropout=0.1,
                       num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`62`)
        chunk_size (int): The sampling rate of EEG signals. (default: :obj:`64`)
        d_model (int): The dimension of the embedding model. (default: :obj:`16`)
        num_heads (int): The number of attention heads. (default: :obj:`8`)
        attn_dropout (float): The dropout rate for attention layers. (default: :obj:`0.1`)
        num_classes (int): The number of classes. (default: :obj:`2`)
    '''

    def __init__(self,
                 num_electrodes: int = 62,
                 chunk_size: int = 64,
                 d_model: int = 16,
                 num_heads: int = 8,
                 attn_dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.num_classes = num_classes

        self.token_embedding = TokenEmbedding(num_electrodes, d_model)
        self.stack1 = AttnRefine(d_model, num_heads, attn_dropout)
        self.stack2 = AttnRefine(d_model, num_heads, attn_dropout)

        self.flatten = nn.Flatten()
        self.out = nn.Linear(8, num_classes)

    def feature_dim(self) -> int:
        with torch.no_grad():
            mock_eeg = torch.zeros(
                1, 1, self.num_electrodes, self.chunk_size)

            x_src = self.token_embedding(mock_eeg)
            x_src1, new_src1 = self.stack1(x_src)
            x_src2, new_src2 = self.stack2(x_src1)

            out = torch.cat([new_src1, new_src2], -1)
            out = self.flatten(out)

            return out.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 64, 64]`. Here, :obj:`n` corresponds to the batch size, the first :obj:`64` corresponds to :obj:`num_electrodes`, and the second :obj:`64` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[size of batch, number of classes]: The predicted probability that the samples belong to the classes.
        '''
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        out = torch.cat(new_x, -1)
        out = self.flatten(out)
        out = self.out(out)
        return out
