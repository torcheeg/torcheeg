import math
from collections import OrderedDict

import torch
from torch import nn

from ..layers import FourierResample
from .steegformer import STEEGFormer
from ._utils import _positive_int, _probability


class _STEEGFormerDownstream(nn.Module):
    def __init__(self, backbone, channel_names, chunk_size, sampling_rate, normalize_signal,
                 std_correction, std_floor, normalize_features, freeze_encoder):
        super().__init__()
        if not isinstance(backbone, STEEGFormer):
            raise ValueError('backbone must be a STEEGFormer instance.')
        _positive_int('chunk_size', chunk_size)
        _positive_int('sampling_rate', sampling_rate)
        for name, value in [('normalize_signal', normalize_signal), ('normalize_features', normalize_features),
                            ('freeze_encoder', freeze_encoder)]:
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean.')
        if isinstance(std_correction, bool) or not isinstance(std_correction, int) or std_correction not in (0, 1):
            raise ValueError('std_correction must be zero or one.')
        if (isinstance(std_floor, bool) or not isinstance(std_floor, (int, float))
                or not math.isfinite(std_floor) or std_floor <= 0):
            raise ValueError('std_floor must be finite and positive in the supplied signal units.')
        if normalize_signal and chunk_size <= std_correction:
            raise ValueError('Signal normalization needs more samples than std_correction.')
        if chunk_size * 128 % sampling_rate:
            raise ValueError('Window duration must map to an exact integer sample count at 128 Hz.')
        native_samples = chunk_size * 128 // sampling_rate
        if not native_samples or native_samples % backbone.patch_size or native_samples // backbone.patch_size > backbone.max_patches:
            raise ValueError('Resampled window must contain complete patches within backbone.max_patches.')
        indices = backbone.channel_indices(channel_names)
        if indices.max().item() >= backbone.num_channel_embeddings:
            raise ValueError('Channel names exceed the configured backbone channel table.')
        self.backbone = backbone
        self.channel_names = tuple(channel_names)
        self.num_electrodes = len(channel_names)
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.native_samples = native_samples
        self.num_patches = native_samples // backbone.patch_size
        self.normalize_signal = normalize_signal
        self.std_correction = std_correction
        self.std_floor = std_floor
        self.normalize_features = normalize_features
        self.freeze_encoder = freeze_encoder
        self.register_buffer('channel_indices', indices.to(device=next(backbone.parameters()).device))
        self.resample = FourierResample(native_samples)
        if freeze_encoder:
            backbone.requires_grad_(False)
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        return self

    def preprocess(self, x):
        r'''Validate raw windows, optionally standardize, then resample the complete window.

        Input is [batch, electrodes, chunk_size] at the explicit sampling_rate.
        Optional per-channel/window z-scoring happens BEFORE Fourier resampling.
        std_correction chooses population (0) or sample (1) standard deviation;
        std_floor is expressed in the supplied input units. Low-precision
        statistics are computed in float32 and cast back to the signal dtype.
        '''
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3
                or x.shape[0] < 1 or x.shape[1:] != (self.num_electrodes, self.chunk_size)):
            raise ValueError('Input must be real floating [batch, electrodes, chunk_size] with the configured source shape.')
        if self.normalize_signal:
            values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            mean = values.mean(dim=-1, keepdim=True)
            std = values.std(dim=-1, keepdim=True, unbiased=self.std_correction == 1).clamp_min(self.std_floor)
            x = ((values - mean) / std).to(dtype=x.dtype)
        return self.resample(x)

    def feature(self, x, flatten=False):
        r'''Return [batch, electrodes, patches, width] features, excluding CLS.

        normalize_features selects the backbone's pre-norm (False) or native
        pretraining post-norm (True) representation. flatten=True flattens
        this channel-major grid from axis one. Patch count comes from the
        validated physical duration, not a lossy reverse-rate calculation.
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        features = self.backbone.forward_features(self.preprocess(x), self.channel_indices,
            normalize=self.normalize_features).permute(0, 2, 1, 3)
        return features.flatten(1) if flatten else features

    @classmethod
    def from_pretrained(cls, checkpoint_path, backbone_kwargs, prefix='', exclude_decoder=False,
                        allow_namespace=False, **kwargs):
        r'''Build a task from a complete local native encoder state.

        Supply explicit backbone_kwargs and task/preprocessing settings.
        Native decoder exclusion and metadata opt-in follow the encoder's
        safe loader. No download or partial encoder restoration occurs.
        The task head starts from random weights. To restore a trained task,
        rebuild all configurations and load its complete state strictly.
        '''
        if not isinstance(backbone_kwargs, dict):
            raise ValueError('backbone_kwargs must be a dictionary of STEEGFormer settings.')
        backbone = STEEGFormer.from_pretrained(checkpoint_path, prefix=prefix,
            exclude_decoder=exclude_decoder, allow_namespace=allow_namespace, **backbone_kwargs)
        return cls(backbone=backbone, **kwargs)


class _STEEGFormerPrediction(_STEEGFormerDownstream):
    def __init__(self, backbone, channel_names, chunk_size, outputs, sampling_rate, dropout,
                 normalize_signal, std_correction, std_floor, normalize_features, freeze_encoder):
        _positive_int('outputs', outputs)
        _probability('dropout', dropout)
        super().__init__(backbone, channel_names, chunk_size, sampling_rate, normalize_signal,
                         std_correction, std_floor, normalize_features, freeze_encoder)
        self.task_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)), ('output', nn.Linear(backbone.hid_channels, outputs))
        ])).to(next(backbone.parameters()))

    def forward(self, x):
        return self.task_head(self.feature(x).mean(dim=(1, 2)))


class STEEGFormerClassifier(_STEEGFormerPrediction):
    r'''Classify explicitly sampled raw EEG with ST-EEGFormer mean pooling.

    Return unnormalized [batch, num_classes] logits. The native downstream
    head applies dropout and a linear map to the mean of pre-norm patch
    tokens, excluding CLS. Set normalize_features=True explicitly for the
    archived workspace's mean-of-post-norm representation.

    Supply the ordered channel_names and source sampling_rate. The complete
    raw window is Fourier resampled to 128 Hz while preserving its exact
    physical duration, then divided into complete backbone patches. There
    is no silent sample truncation or sensor relabelling. Signal z-scoring
    is optional and explicitly precedes resampling. Filtering, reference
    selection and input units remain part of the data pipeline.

    .. code-block:: python

        import torch
        from torcheeg.models import STEEGFormer, STEEGFormerClassifier

        model = STEEGFormerClassifier(STEEGFormer(), ['Fp1', 'Fp2', 'Cz'],
                                      chunk_size=800, sampling_rate=200)
        logits = model(torch.randn(2, 3, 800))

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (STEEGFormer): Explicitly configured encoder with its complete native state.
        channel_names (list of str): Ordered pretrained names matching the input electrodes.
        chunk_size (int): Raw samples in each source window.
        num_classes (int): Output class count. (default: :obj:`2`)
        sampling_rate (int): Source sampling rate in Hz; duration must map exactly to complete 128 Hz patches. (default: :obj:`128`)
        dropout (float): Task-head dropout probability. (default: :obj:`0.1`)
        normalize_signal (bool): Per-channel/window z-score before resampling. (default: :obj:`False`)
        std_correction (int): Zero for population std, one for sample std. (default: :obj:`0`)
        std_floor (float): Positive std floor in the supplied signal units. (default: :obj:`1e-6`)
        normalize_features (bool): Apply the encoder's final norm before pooling. False matches the official downstream global-pooling path. (default: :obj:`False`)
        freeze_encoder (bool): Freeze encoder parameters and keep it in evaluation mode while retaining input gradients. Otherwise preserve existing parameter flags. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, num_classes=2, sampling_rate=128,
                 dropout=.1, normalize_signal=False, std_correction=0, std_floor=1e-6,
                 normalize_features=False, freeze_encoder=False):
        super().__init__(backbone, channel_names, chunk_size, num_classes, sampling_rate, dropout,
                         normalize_signal, std_correction, std_floor, normalize_features, freeze_encoder)
        self.num_classes = num_classes


class STEEGFormerRegression(_STEEGFormerPrediction):
    r'''Predict continuous targets with the ST-EEGFormer mean-pooling head.

    Input is [batch, electrodes, chunk_size]; output is [batch, num_outputs],
    retaining the scalar target axis. No output activation or target scaling
    is applied. Input, preprocessing, modes and local encoder loading follow
    STEEGFormerClassifier. The built-in RegressorTrainer currently supports
    scalar targets; use an explicit multi-output loss for num_outputs > 1.

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (STEEGFormer): Configured encoder.
        channel_names (list of str): Ordered pretrained electrode names.
        chunk_size (int): Source samples per window.
        num_outputs (int): Continuous output dimensions. (default: :obj:`1`)
        sampling_rate (int): Explicit source Hz, with exact 128 Hz duration mapping. (default: :obj:`128`)
        dropout (float): Task-head dropout. (default: :obj:`0.1`)
        normalize_signal (bool): Per-channel/window z-score before resampling. (default: :obj:`False`)
        std_correction (int): Population (0) or sample (1) standard deviation. (default: :obj:`0`)
        std_floor (float): Std floor in the supplied signal units. (default: :obj:`1e-6`)
        normalize_features (bool): Apply the encoder's final norm. (default: :obj:`False`)
        freeze_encoder (bool): Freeze the encoder and keep it in evaluation mode. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, num_outputs=1, sampling_rate=128,
                 dropout=.1, normalize_signal=False, std_correction=0, std_floor=1e-6,
                 normalize_features=False, freeze_encoder=False):
        super().__init__(backbone, channel_names, chunk_size, num_outputs, sampling_rate, dropout,
                         normalize_signal, std_correction, std_floor, normalize_features, freeze_encoder)
        self.num_outputs = num_outputs


class STEEGFormerEmbedding(_STEEGFormerDownstream):
    r'''Extract ST-EEGFormer embeddings without a learned task head.

    Default mean pooling returns [batch, width], matching the archived
    wrapper's pooled output shape. pooling='none' returns the channel-major
    [batch, electrodes, patches, width] grid; 'flatten' flattens that grid.
    Preprocessing and feature normalization are explicit and follow the
    prediction wrappers. Random initial weights are not pretrained features.

    .. automethod:: preprocess
    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (STEEGFormer): Configured encoder with complete native state.
        channel_names (list of str): Ordered pretrained electrode names.
        chunk_size (int): Source samples per window.
        pooling (str): 'mean', 'none', or channel-major 'flatten'. (default: :obj:`'mean'`)
        sampling_rate (int): Explicit source Hz, preserving physical duration. (default: :obj:`128`)
        normalize_signal (bool): Per-channel/window z-score before resampling. (default: :obj:`False`)
        std_correction (int): Population (0) or sample (1) std. (default: :obj:`0`)
        std_floor (float): Std floor in supplied signal units. (default: :obj:`1e-6`)
        normalize_features (bool): Apply the encoder's final norm. True restores the archived post-norm representation. (default: :obj:`False`)
        freeze_encoder (bool): Freeze the encoder and keep it in evaluation mode while retaining input gradients. (default: :obj:`False`)
    '''
    def __init__(self, backbone, channel_names, chunk_size, pooling='mean', sampling_rate=128,
                 normalize_signal=False, std_correction=0, std_floor=1e-6,
                 normalize_features=False, freeze_encoder=False):
        if pooling not in ('mean', 'none', 'flatten'):
            raise ValueError('pooling must be mean, none or flatten.')
        super().__init__(backbone, channel_names, chunk_size, sampling_rate, normalize_signal,
                         std_correction, std_floor, normalize_features, freeze_encoder)
        self.pooling = pooling

    def forward(self, x):
        features = self.feature(x)
        if self.pooling == 'mean':
            return features.mean(dim=(1, 2))
        return features.flatten(1) if self.pooling == 'flatten' else features
