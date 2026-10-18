# Adapted from the MIT-licensed SleepDG implementation by Jiquan Wang.
# See LICENSES/SleepDG.txt and LICENSES/SleepDG-NOTICE.txt.
from collections import OrderedDict

import torch
from torch import nn

from ._utils import _positive_int, _probability


class _Mlp(nn.Sequential):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden, width)
        self.dropout_2 = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class _Block(nn.Module):
    def __init__(self, width, hidden, heads, dropout, attention_dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = _Mlp(width, hidden, dropout)

    def forward(self, x):
        normalized = self.ln_1(x)
        update = self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        x = x + self.dropout(update)
        return x + self.mlp(self.ln_2(x))


class _Transformer(nn.Module):
    def __init__(self, length, width, hidden, depth, heads, dropout, attention_dropout):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, length, width).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            OrderedDict(
                (f'encoder_layer_{i}', _Block(width, hidden, heads, dropout, attention_dropout))
                for i in range(depth)
            )
        )
        self.ln = nn.LayerNorm(width, eps=1e-6)

    def forward(self, x):
        return self.ln(self.layers(self.dropout(x + self.pos_embedding)))


class _EpochEncoder(nn.Module):
    def __init__(self, electrodes, width, dropout):
        super().__init__()
        layers = []
        for index, (input_width, output_width) in enumerate(
            zip((electrodes, 64, 128, 256), (64, 128, 256, width))
        ):
            layers.extend(
                [
                    nn.Conv1d(
                        input_width,
                        output_width,
                        49 if index == 0 else 9,
                        stride=6 if index == 0 else 1,
                        padding=24 if index == 0 else 4,
                        bias=False,
                    ),
                    nn.BatchNorm1d(output_width),
                    nn.GELU(),
                    nn.MaxPool1d(9, stride=2, padding=4),
                ]
            )
            if index == 0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Keep the sample axis when batch * sequence or hidden width equals one.
        return self.avg(self.encoder(x)).squeeze(-1)


class _Encoder(nn.Module):
    def __init__(self, electrodes, length, width, hidden, depth, heads, dropout, attention_dropout):
        super().__init__()
        self.epoch_encoder = _EpochEncoder(electrodes, width, dropout)
        self.seq_encoder = _Transformer(
            length, width, hidden, depth, heads, dropout, attention_dropout
        )
        self.fc_mu = nn.Linear(width, width)

    def forward(self, x):
        batch, length, electrodes, samples = x.shape
        epochs = self.epoch_encoder(x.reshape(batch * length, electrodes, samples))
        return self.fc_mu(self.seq_encoder(epochs.reshape(batch, length, -1)))


class _Decoder(nn.Module):
    def __init__(self, electrodes, width, dropout):
        super().__init__()
        layers = []
        for input_width, output_width, kernel, stride, padding in (
            (width, width, 5, 5, 0),
            (width, 256, 10, 2, 4),
            (256, 128, 10, 2, 4),
            (128, 64, 10, 2, 4),
            (64, 64, 5, 5, 0),
        ):
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        input_width, output_width, kernel, stride, padding, bias=False
                    ),
                    nn.BatchNorm1d(output_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.ConvTranspose1d(64, electrodes, 49, 15, 17, bias=False))
        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        batch, length, width = x.shape
        return self.upsample(x.reshape(batch * length, width, 1)).reshape(batch, length, -1, 3000)


class _Autoencoder(nn.Module):
    def __init__(self, electrodes, length, width, hidden, depth, heads, dropout, attention_dropout):
        super().__init__()
        self.encoder = _Encoder(
            electrodes, length, width, hidden, depth, heads, dropout, attention_dropout
        )
        self.decoder = _Decoder(electrodes, width, dropout)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features), features


class SleepDG(nn.Module):
    r"""SleepDG sequence classifier and reconstruction model from Wang et al. (AAAI 2024).

    Supply [batch, sequence_length, num_electrodes, 3000] real floating
    epochs. Native input contains 20 consecutive 30-second epochs at 100 Hz,
    using EEG and EOG in the preprocessing pipeline's explicit channel order.
    Do not cross recording, subject or partition boundaries when making a
    sequence. Filtering, reference, units and normalization belong to the
    data pipeline; the model does not infer a montage or resample signals.

    Four convolution/BatchNorm/GELU/pooling stages encode each epoch, followed
    by adaptive temporal averaging. A learned-position Transformer operates
    ACROSS EPOCHS. The projected fc_mu features feed a per-epoch classifier
    and six transposed convolutions reconstruct each 3000-sample epoch.
    Native state paths, including ae.encoder, ae.decoder and BatchNorm
    buffers, are retained. Sequence length, electrode count and hidden width
    may be configured; a native state requires its matching configuration.

    forward returns unnormalized [batch, sequence_length, num_classes]
    logits through the encoder only. forward_all returns logits,
    reconstruction and features, matching the author's full model path.
    Decoder BatchNorm/dropout runs only when reconstruction is requested.
    This architecture alone does not apply the epoch/sequence domain
    alignment or constitute the complete domain-generalization training
    protocol. Flatten logits and labels consistently for cross-entropy;
    the ordinary ClassifierTrainer does not infer this sequence contract.

    - Paper: Wang J, Zhao S, Jiang H, et al. Generalizable Sleep Staging via Multi-Level Domain Alignment. AAAI 2024.
    - URL: https://arxiv.org/abs/2401.05363
    - Related Project: https://github.com/wjq-learning/SleepDG

    .. code-block:: python

        from torcheeg.models import SleepDG
        import torch

        model = SleepDG()
        x = torch.randn(2, 20, 2, 3000)
        logits = model(x)
        logits, reconstruction, features = model.forward_all(x)

    .. automethod:: feature
    .. automethod:: forward_all
    .. automethod:: feature_dim

    Args:
        num_electrodes (int): Ordered input channels; native EEG/EOG pair. (default: :obj:`2`)
        chunk_size (int): Samples per epoch; native decoder requires 3000. (default: :obj:`3000`)
        sequence_length (int): Consecutive epochs per sequence. (default: :obj:`20`)
        num_classes (int): Per-epoch sleep stages. (default: :obj:`5`)
        hid_channels (int): Epoch, sequence and projected feature width. (default: :obj:`512`)
        num_layers (int): Sequence Transformer depth. (default: :obj:`1`)
        num_heads (int): Heads dividing hid_channels. (default: :obj:`8`)
        mlp_channels (int): Sequence MLP width. (default: :obj:`512`)
        dropout (float): Encoder, sequence and decoder dropout. (default: :obj:`0.1`)
        attention_dropout (float, optional): Attention probability dropout, or dropout. (default: :obj:`None`)
    """

    def __init__(
        self,
        num_electrodes=2,
        chunk_size=3000,
        sequence_length=20,
        num_classes=5,
        hid_channels=512,
        num_layers=1,
        num_heads=8,
        mlp_channels=512,
        dropout=0.1,
        attention_dropout=None,
    ):
        super().__init__()
        for name, value in dict(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            sequence_length=sequence_length,
            num_classes=num_classes,
            hid_channels=hid_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_channels=mlp_channels,
        ).items():
            _positive_int(name, value)
        if chunk_size != 3000:
            raise ValueError('The native SleepDG decoder reconstructs 3000-sample epochs.')
        if hid_channels % num_heads:
            raise ValueError('hid_channels must be divisible by num_heads.')
        _probability('dropout', dropout)
        attention_dropout = dropout if attention_dropout is None else attention_dropout
        _probability('attention_dropout', attention_dropout)
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.ae = _Autoencoder(
            num_electrodes,
            sequence_length,
            hid_channels,
            mlp_channels,
            num_layers,
            num_heads,
            dropout,
            attention_dropout,
        )
        self.classifier = nn.Linear(hid_channels, num_classes)

    def _validate_input(self, x):
        if (
            not isinstance(x, torch.Tensor)
            or not x.is_floating_point()
            or x.ndim != 4
            or x.shape[0] < 1
            or x.shape[1:] != (self.sequence_length, self.num_electrodes, self.chunk_size)
        ):
            raise ValueError(
                'Input must be real floating [batch, sequence_length, num_electrodes, 3000] epochs.'
            )

    def feature(self, x):
        r"""Return projected native fc_mu features [batch, sequence_length, hid_channels]."""
        self._validate_input(x)
        return self.ae.encoder(x)

    def feature_dim(self):
        r"""Return the per-epoch projected feature width."""
        return self.hid_channels

    def forward(self, x):
        return self.classifier(self.feature(x))

    def forward_all(self, x):
        r"""Return per-epoch logits, reconstructed epochs and projected features.

        Shapes are [B, S, classes], [B, S, electrodes, 3000] and [B, S, width].
        This is the author's full forward path, including decoder training
        modes and BatchNorm buffers. It does not compute domain alignment.
        """
        self._validate_input(x)
        reconstruction, features = self.ae(x)
        return self.classifier(features), reconstruction, features
