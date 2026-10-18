from collections import OrderedDict

import torch
from torch import nn

from ..layers import FourierResample
from .labram_encoder import LaBraMEncoder
from ._utils import _positive_int, _probability


class _LaBraMDownstream(nn.Module):
    def __init__(self, backbone, channel_names, chunk_size, sampling_rate,
                 resample_mode, normalize_features, freeze_encoder):
        super().__init__()
        if not isinstance(backbone, LaBraMEncoder):
            raise ValueError('backbone must be a LaBraMEncoder instance.')
        _positive_int('chunk_size', chunk_size)
        _positive_int('sampling_rate', sampling_rate)
        if resample_mode not in ('window', 'patch'):
            raise ValueError('resample_mode must be window or patch.')
        if not isinstance(normalize_features, bool) or not isinstance(freeze_encoder, bool):
            raise ValueError('normalize_features and freeze_encoder must be booleans.')
        if chunk_size * 200 % sampling_rate:
            raise ValueError('Physical duration must map to an exact integer sample count at 200 Hz.')
        native_samples = chunk_size * 200 // sampling_rate
        if not native_samples or native_samples % backbone.patch_size:
            raise ValueError('Resampled window must contain complete native patches without truncation.')
        patches = native_samples // backbone.patch_size
        if patches > backbone.max_patches and not backbone.interpolate_time:
            raise ValueError('Window exceeds the native temporal table; explicitly configure backbone interpolation.')
        if resample_mode == 'patch' and chunk_size % patches:
            raise ValueError('Independent patch resampling requires equal complete source segments.')
        indices = backbone.channel_indices(channel_names)
        self.backbone = backbone
        self.channel_names = tuple(channel_names)
        self.num_electrodes = len(channel_names)
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.native_samples = native_samples
        self.num_patches = patches
        self.resample_mode = resample_mode
        self.normalize_features = normalize_features
        self.freeze_encoder = freeze_encoder
        self.feature_dim = self.num_electrodes * patches * backbone.hid_channels
        self.register_buffer('channel_indices', indices.to(device=next(backbone.parameters()).device))
        self.resample = FourierResample(native_samples if resample_mode == 'window' else backbone.patch_size)
        if freeze_encoder:
            backbone.requires_grad_(False)
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        return self

    def preprocess(self, x):
        r'''Validate source [batch, electrodes, chunk_size] and resample to 200 Hz.

        window mode treats the complete window as one Fourier period. patch
        mode independently resamples equal source segments, preserving the
        archived workspace's operation. These modes can differ at boundaries.
        Neither applies filtering, amplitude scaling or signal z-scoring.
        '''
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3
                or x.shape[0] < 1 or x.shape[1:] != (self.num_electrodes, self.chunk_size)):
            raise ValueError('Input must be real floating [batch, electrodes, chunk_size] with the configured source shape.')
        if self.resample_mode == 'window':
            return self.resample(x)
        segments = x.reshape(x.shape[0], self.num_electrodes, self.num_patches, -1)
        return self.resample(segments).reshape(x.shape[0], self.num_electrodes, self.native_samples)

    def feature(self, x, flatten=False):
        r'''Return [batch, electrodes, patches, width], excluding CLS.

        normalize_features=True uses the complete native encoder's learned
        final norm on each token. False exposes raw final-block tokens.
        flatten=True preserves channel-major, temporal-minor feature order.
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        features = self.backbone.forward_features(self.preprocess(x), self.channel_indices,
            normalize=self.normalize_features)
        return features.flatten(1) if flatten else features

    @classmethod
    def from_pretrained(cls, checkpoint_path, backbone_kwargs, prefix='',
                        exclude_pretraining=False, allow_training_metadata=False, **kwargs):
        r'''Restore a complete local encoder and initialize a fresh task head.

        backbone_kwargs explicitly configures LaBraMEncoder. Prefix selection,
        removal of pretraining-only tensors and scoped known-metadata opt-in
        follow its safe loader. No download, partial restoration or unsafe
        pickle fallback occurs. For a trained task, rebuild both constructor
        configurations and strictly restore its complete tensor state.
        '''
        if not isinstance(backbone_kwargs, dict):
            raise ValueError('backbone_kwargs must be a dictionary of LaBraMEncoder settings.')
        backbone = LaBraMEncoder.from_pretrained(checkpoint_path, prefix=prefix,
            exclude_pretraining=exclude_pretraining, allow_training_metadata=allow_training_metadata, **backbone_kwargs)
        return cls(backbone=backbone, **kwargs)


class _LaBraMPrediction(_LaBraMDownstream):
    def __init__(self, backbone, channel_names, chunk_size, outputs, sampling_rate,
                 resample_mode, normalize_features, pooling, head_channels, dropout, freeze_encoder):
        _positive_int('outputs', outputs)
        _probability('dropout', dropout)
        if pooling not in ('flatten', 'mean'):
            raise ValueError('pooling must be flatten or mean.')
        if head_channels is not None:
            if not isinstance(head_channels, (list, tuple)):
                raise ValueError('head_channels must be a tuple/list of positive integers, or None.')
            for value in head_channels:
                _positive_int('head_channels entry', value)
        super().__init__(backbone, channel_names, chunk_size, sampling_rate,
                         resample_mode, normalize_features, freeze_encoder)
        if head_channels is None:
            head_channels = (max(4, self.num_patches) * backbone.hid_channels, backbone.hid_channels)
        self.pooling = pooling
        self.head_channels = tuple(head_channels)
        width = self.feature_dim if pooling == 'flatten' else backbone.hid_channels
        layers = OrderedDict()
        for index, hidden in enumerate(head_channels):
            layers[f'linear{index}'] = nn.Linear(width, hidden)
            layers[f'activation{index}'] = nn.ELU()
            layers[f'dropout{index}'] = nn.Dropout(dropout)
            width = hidden
        layers['output'] = nn.Linear(width, outputs)
        self.task_head = nn.Sequential(layers).to(next(backbone.parameters()))

    def forward(self, x):
        features = self.feature(x)
        pooled = features.flatten(1) if self.pooling == 'flatten' else features.mean(dim=(1, 2))
        return self.task_head(pooled)


class LaBraMClassifier(_LaBraMPrediction):
    r'''Classify explicitly sampled raw EEG with LaBraM patch-token features.

    Return unnormalized [batch, num_classes] logits. By default flatten the
    channel-major patch grid and apply max(4, patches)*H -> H -> classes,
    with ELU and configurable dropout after each hidden layer. This retains
    the archived workspace's task-head topology. H is backbone.hid_channels.
    head_channels=() selects a linear head with no hidden-layer dropout.

    Supply source sampling_rate, chunk_size and ordered channel_names.
    Preserve exact physical duration when Fourier-resampling to native 200 Hz.
    resample_mode='window' treats the full window as one period; 'patch'
    explicitly retains independent per-patch resampling. Complete patches
    are required in both modes. Filtering, EEG reference and units belong to
    the data pipeline; native fine-tuning scales microvolts by 1/100.

    normalize_features=True applies the native learned norm PER TOKEN before
    flattening or mean pooling, corresponding to the archived all-token path
    with complete encoder norm restoration. It differs from norm AFTER mean
    pooling in LaBraMEncoder.forward and from a fresh native fc_norm.

    .. code-block:: python

        import torch
        from torcheeg.models import LaBraMEncoder, LaBraMClassifier

        model = LaBraMClassifier(LaBraMEncoder(), ['Fp1', 'C3', 'C4'],
                                  chunk_size=512, sampling_rate=128)
        logits = model(torch.randn(2, 3, 512))

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (LaBraMEncoder): Complete explicitly configured native encoder.
        channel_names (list of str): Ordered known native electrode identities.
        chunk_size (int): Source samples per window.
        num_classes (int): Output classes. (default: :obj:`2`)
        sampling_rate (int): Source Hz; physical duration must map to complete 200 Hz patches. (default: :obj:`200`)
        resample_mode (str): Whole 'window' or independent 'patch' resampling. (default: :obj:`'window'`)
        normalize_features (bool): Apply the learned native norm separately to each token. (default: :obj:`True`)
        pooling (str): 'flatten' or mean over channel/patch axes. (default: :obj:`'flatten'`)
        head_channels (tuple, optional): Hidden widths; None means (max(4,patches)*H,H), () means linear. (default: :obj:`None`)
        dropout (float): Dropout after each hidden ELU; no dropout for an empty hidden tuple. (default: :obj:`0.1`)
        freeze_encoder (bool): Freeze encoder parameters and keep it in evaluation mode while retaining input gradients. Otherwise preserve existing flags. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, num_classes=2, sampling_rate=200,
                 resample_mode='window', normalize_features=True, pooling='flatten',
                 head_channels=None, dropout=.1, freeze_encoder=False):
        super().__init__(backbone, channel_names, chunk_size, num_classes, sampling_rate,
                         resample_mode, normalize_features, pooling, head_channels, dropout, freeze_encoder)
        self.num_classes = num_classes


class LaBraMRegression(_LaBraMPrediction):
    r'''Predict continuous [batch, num_outputs] targets with LaBraM features.

    The scalar target axis remains present. No output activation or target
    scaling is applied. Sampling, feature normalization, pooling and head
    behavior follow LaBraMClassifier. The built-in RegressorTrainer supports
    scalar targets; use an explicit loss for multiple output dimensions.

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (LaBraMEncoder): Complete native encoder.
        channel_names (list of str): Ordered native channel names.
        chunk_size (int): Source samples per window.
        num_outputs (int): Continuous target dimensions. (default: :obj:`1`)
        sampling_rate (int): Explicit source Hz. (default: :obj:`200`)
        resample_mode (str): 'window' or independent 'patch'. (default: :obj:`'window'`)
        normalize_features (bool): Apply native final norm per token. (default: :obj:`True`)
        pooling (str): 'flatten' or 'mean'. (default: :obj:`'flatten'`)
        head_channels (tuple, optional): Hidden widths, or () for linear. None means (max(4,patches)*H,H). (default: :obj:`None`)
        dropout (float): Dropout after hidden ELU activations. (default: :obj:`0.1`)
        freeze_encoder (bool): Freeze the encoder and keep it in evaluation mode. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, num_outputs=1, sampling_rate=200,
                 resample_mode='window', normalize_features=True, pooling='flatten',
                 head_channels=None, dropout=.1, freeze_encoder=False):
        super().__init__(backbone, channel_names, chunk_size, num_outputs, sampling_rate,
                         resample_mode, normalize_features, pooling, head_channels, dropout, freeze_encoder)
        self.num_outputs = num_outputs


class LaBraMEmbedding(_LaBraMDownstream):
    r'''Extract ordered LaBraM features without a learned task head.

    Return flattened [batch, electrodes*patches*width] features by default,
    or a channel-major [batch, electrodes, patches, width] grid. Sampling and
    feature normalization follow LaBraMClassifier; no signal amplitude
    normalization, learned projection or label fitting is performed here.
    Random weights do not produce pretrained representations.

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (LaBraMEncoder): Complete native encoder.
        channel_names (list of str): Ordered known native identities.
        chunk_size (int): Source samples per window.
        flatten (bool): Flatten features from axis one. (default: :obj:`True`)
        sampling_rate (int): Explicit source Hz. (default: :obj:`200`)
        resample_mode (str): Whole 'window' or independent 'patch' resampling. (default: :obj:`'window'`)
        normalize_features (bool): Apply learned final norm per token. (default: :obj:`True`)
        freeze_encoder (bool): Freeze encoder and retain evaluation mode/input gradients. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, flatten=True, sampling_rate=200,
                 resample_mode='window', normalize_features=True, freeze_encoder=False):
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        super().__init__(backbone, channel_names, chunk_size, sampling_rate,
                         resample_mode, normalize_features, freeze_encoder)
        self.flatten = flatten

    def forward(self, x):
        return self.feature(x, flatten=self.flatten)
