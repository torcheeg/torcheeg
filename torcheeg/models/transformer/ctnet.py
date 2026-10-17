# CTNet architecture: Zhao et al. (2024), https://github.com/snailpt/CTNet.
# Copyright (c) 2024 snailpt. See LICENSES/CTNet.txt for the MIT notice.

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SamePadConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size[-1] - 1
        return super().forward(F.pad(x, (total // 2, total - total // 2, 0, 0)))


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding module for CTNet.

    Uses temporal convolution, depth-wise spatial convolution, and spatial convolution
    followed by average pooling to create patch embeddings.
    """

    def __init__(self,
                 num_electrodes: int = 22,
                 f1: int = 16,
                 d: int = 2,
                 temporal_kernel: int = 64,
                 spatial_kernel: int = 16,
                 pool_size1: int = 8,
                 pool_size2: int = 8,
                 dropout: float = 0.5):
        super(PatchEmbedding, self).__init__()

        f2 = d * f1

        self.cnn_module = nn.Sequential(
            # Temporal convolution
            _SamePadConv2d(1, f1, (1, temporal_kernel),
                           (1, 1), padding=0, bias=False),
            nn.BatchNorm2d(f1),
            # Channel depth-wise convolution
            nn.Conv2d(f1, f2, (num_electrodes, 1), (1, 1),
                      groups=f1, padding=0, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # Average pooling 1
            nn.AvgPool2d((1, pool_size1)),
            nn.Dropout(dropout),
            # Spatial convolution
            _SamePadConv2d(f2, f2, (1, spatial_kernel), padding=0, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # Average pooling 2
            nn.AvgPool2d((1, pool_size2)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_module(x)
        x = self.projection(x).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for CTNet."""

    def __init__(self,
                 hid_channels: int,
                 max_length: int = 1000,
                 dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_length, hid_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.encoding[:, :x.shape[1], :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for CTNet."""

    def __init__(self,
                 hid_channels: int,
                 heads: int,
                 dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        head_dim = self.hid_channels // self.heads
        projections = [projection(x).reshape(batch_size, num_tokens, self.heads, head_dim)
                       .transpose(1, 2) for projection in (self.queries, self.keys, self.values)]
        query, key, value = projections
        scores = query @ key.transpose(-2, -1)
        # CTNet scales by the embedding width, not the per-head width.
        scores = scores / math.sqrt(self.hid_channels)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.att_drop(F.softmax(scores, dim=-1))
        features = (weights @ value).transpose(1, 2).reshape(batch_size, num_tokens, self.hid_channels)
        return self.projection(features)


class FeedForwardBlock(nn.Sequential):
    """Feed-forward block for CTNet Transformer encoder."""

    def __init__(self,
                 hid_channels: int,
                 expansion: int = 4,
                 dropout: float = 0.5):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )


class ResidualAdd(nn.Module):
    """Residual connection with layer normalization for CTNet."""

    def __init__(self, fn: nn.Module, hid_channels: int, dropout: float):
        super(ResidualAdd, self).__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_channels)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.dropout(res) + x)
        return out


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block for CTNet."""

    def __init__(self,
                 hid_channels: int,
                 heads: int = 2,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = ResidualAdd(
            MultiHeadAttention(hid_channels, heads, dropout),
            hid_channels,
            dropout
        )
        self.feed_forward = ResidualAdd(
            FeedForwardBlock(
                hid_channels, expansion=forward_expansion, dropout=forward_dropout),
            hid_channels,
            dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for CTNet."""

    def __init__(self,
                 depth: int,
                 hid_channels: int,
                 heads: int = 2,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                hid_channels=hid_channels,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                forward_dropout=forward_dropout
            ) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationHead(nn.Sequential):
    """Classification head for CTNet."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout: float = 0.5):
        super(ClassificationHead, self).__init__(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )


class CTNet(nn.Module):
    r'''
    CTNet: A Convolutional Transformer Network for EEG-based Motor Imagery Classification.

    CTNet combines the local feature extraction capability of CNNs with the global
    feature learning ability of Transformers. It first uses an EEGNet-like convolutional
    module to extract local temporal and spatial features, then employs a Transformer
    encoder with multi-head attention to capture global dependencies. A key feature is
    the residual connection between CNN features and Transformer output.

    - Paper: Zhao W, Jiang X, Zhang B, et al. CTNet: a convolutional transformer network
      for EEG-based motor imagery classification[J]. Scientific reports, 2024, 14(1): 20237.
    - URL: https://www.nature.com/articles/s41598-024-71118-7
    - Related Project: https://github.com/snailpt/CTNet

    .. code-block:: python

        from torcheeg.models import CTNet

        model = CTNet(num_electrodes=22,
                      chunk_size=1000,
                      num_classes=4)

        # Input shape: (batch_size, num_electrodes, chunk_size)
        x = torch.randn(32, 22, 1000)
        output = model(x)  # (32, 4)

    This adapter returns logits directly and provides :obj:`feature` for
    flattened features. Inputs have shape [batch, electrodes, time points];
    no filtering, scaling or windowing is performed inside the model. The
    caller must match the chosen data and evaluation protocol. Defaults
    retain the earlier experimental TorchEEG parameterization and do not
    imply reproduction of a published accuracy.

    No dummy training-mode forward pass is run during initialization, so
    BatchNorm statistics remain unmodified. Experimental checkpoints with
    matching constructor settings retain their parameter names and shapes.
    Author-repository checkpoints use different module names and are not
    loaded automatically. Set :obj:`max_length=100` to match the original
    author's positional-embedding length when validating that configuration.

    Args:
        num_electrodes (int): The number of EEG electrodes/channels. (default: :obj:`22`)
        chunk_size (int): Number of time samples in each EEG chunk. (default: :obj:`1000`)
        f1 (int): Number of temporal filters in the first convolution layer. (default: :obj:`16`)
        d (int): Depth multiplier for depth-wise convolution. (default: :obj:`2`)
        temporal_kernel (int): Kernel size for temporal convolution. (default: :obj:`64`)
        spatial_kernel (int): Kernel size for spatial convolution. (default: :obj:`16`)
        pool_size1 (int): First average pooling kernel size. (default: :obj:`8`)
        pool_size2 (int): Second average pooling kernel size. (default: :obj:`8`)
        depth (int): Number of Transformer encoder layers. (default: :obj:`6`)
        heads (int): Number of attention heads in multi-head attention. (default: :obj:`2`)
        dropout (float): Dropout probability for convolutional layers. (default: :obj:`0.5`)
        transformer_dropout (float): Dropout probability for Transformer layers. (default: :obj:`0.5`)
        forward_expansion (int): Expansion factor for feed-forward network. (default: :obj:`4`)
        forward_dropout (float): Dropout probability for feed-forward network. (default: :obj:`0.5`)
        pos_dropout (float): Dropout probability for positional encoding. (default: :obj:`0.1`)
        cls_dropout (float): Dropout probability for classification head. (default: :obj:`0.5`)
        num_classes (int): Number of output classes. (default: :obj:`4`)
        max_length (int): Capacity of the learnable token-position table.
            Must cover the token count after both pools. (default: :obj:`1000`)
    '''

    def __init__(self,
                 num_electrodes: int = 22,
                 chunk_size: int = 1000,
                 f1: int = 16,
                 d: int = 2,
                 temporal_kernel: int = 64,
                 spatial_kernel: int = 16,
                 pool_size1: int = 8,
                 pool_size2: int = 8,
                 depth: int = 6,
                 heads: int = 2,
                 dropout: float = 0.5,
                 transformer_dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5,
                 pos_dropout: float = 0.1,
                 cls_dropout: float = 0.5,
                 num_classes: int = 4,
                 max_length: int = 1000):
        super(CTNet, self).__init__()

        integer_parameters = {
            "num_electrodes": num_electrodes, "chunk_size": chunk_size,
            "f1": f1, "d": d, "temporal_kernel": temporal_kernel,
            "spatial_kernel": spatial_kernel, "pool_size1": pool_size1,
            "pool_size2": pool_size2, "depth": depth, "heads": heads,
            "forward_expansion": forward_expansion, "num_classes": num_classes,
            "max_length": max_length,
        }
        for name, value in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (f1 * d) % heads:
            raise ValueError("f1 * d must be divisible by heads.")
        num_tokens = chunk_size // pool_size1 // pool_size2
        if not 1 <= num_tokens <= max_length:
            raise ValueError("Pooling must produce between one and max_length tokens.")
        for name, value in {"dropout": dropout, "transformer_dropout": transformer_dropout,
                            "forward_dropout": forward_dropout, "pos_dropout": pos_dropout,
                            "cls_dropout": cls_dropout}.items():
            if not isinstance(value, Real) or isinstance(value, bool) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.f1 = f1
        self.d = d
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel = spatial_kernel
        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.transformer_dropout = transformer_dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        self.pos_dropout = pos_dropout
        self.cls_dropout = cls_dropout
        self.num_classes = num_classes
        self.max_length = max_length

        # Embedding dimension is f1 * d
        self.embed_dim = f1 * d

        # Patch embedding (CNN module)
        self.patch_embedding = PatchEmbedding(
            num_electrodes=num_electrodes,
            f1=f1,
            d=d,
            temporal_kernel=temporal_kernel,
            spatial_kernel=spatial_kernel,
            pool_size1=pool_size1,
            pool_size2=pool_size2,
            dropout=dropout
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            hid_channels=self.embed_dim,
            max_length=max_length,
            dropout=pos_dropout
        )

        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            depth=depth,
            hid_channels=self.embed_dim,
            heads=heads,
            dropout=transformer_dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout
        )

        # Classification head
        self._feature_dim = self._calculate_feature_dim()
        self.classification_head = ClassificationHead(
            in_channels=self._feature_dim,
            num_classes=num_classes,
            dropout=cls_dropout
        )

    def _calculate_feature_dim(self) -> int:
        """Calculate the feature dimension after patch embedding and transformer."""
        return self.embed_dim * (self.chunk_size // self.pool_size1 // self.pool_size2)

    def feature_dim(self) -> int:
        """Return the dimension of the feature vector.

        Returns:
            int: The dimension of the flattened feature vector.
        """
        return self._feature_dim

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input EEG signal.

        Args:
            x (torch.Tensor): EEG signal with shape (batch_size, num_electrodes, chunk_size).

        Returns:
            torch.Tensor: Feature vector with shape (batch_size, feature_dim).
        """
        if x.ndim != 3 or x.shape[1] != self.num_electrodes or x.shape[2] != self.chunk_size:
            raise ValueError("CTNet expects [batch, num_electrodes, chunk_size] input.")
        # Add channel dimension: (batch, electrodes, time) -> (batch, 1, electrodes, time)
        x = x.unsqueeze(1)

        # Patch embedding (CNN module)
        cnn_features = self.patch_embedding(x)

        # Scale and add positional encoding
        cnn_features = cnn_features * math.sqrt(self.embed_dim)
        cnn_features = self.positional_encoding(cnn_features)

        # Transformer encoder
        trans_features = self.transformer_encoder(cnn_features)

        # Residual connection between CNN and Transformer
        features = cnn_features + trans_features

        # Flatten
        features = features.flatten(start_dim=1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Forward pass of CTNet.

        Args:
            x (torch.Tensor): EEG signal with shape (batch_size, num_electrodes, chunk_size).
                For example, (32, 22, 1000) for BCI Competition IV-2a dataset.

        Returns:
            torch.Tensor: Classification logits with shape (batch_size, num_classes).
        '''
        features = self.feature(x)
        output = self.classification_head(features)
        return output
