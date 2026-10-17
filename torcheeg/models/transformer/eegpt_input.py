import math

import torch
from torch import nn
from torch.nn import functional as F

from ..cnn.eegnet import Conv2dWithConstraint
from .eegpt import _positive_int, _probability
from .eegpt_heads import _EEGPTDownstream


class EEGPTResample(nn.Module):
    r'''Differentiable Fourier resampling along the final sample axis.

    Resize the one-sided spectrum and correct the unpaired Nyquist bin when
    the shorter length is even. This follows the real, time-domain,
    unwindowed Fourier-resampling convention of scipy.signal.resample.
    The runtime implementation uses PyTorch only. It treats each input
    window as periodic; discontinuities at its endpoints can cause ringing.

    Input and output use the same floating dtype and device. CPU float16
    and bfloat16 spectra are computed in float32, then converted back.
    Equal lengths return the input directly. No signal units or amplitude
    normalization are changed. The number of samples alone does not declare
    a sampling rate: the caller must preserve and state the window duration.

    - Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPTResample

        # Four seconds at 200 Hz -> four seconds at 256 Hz.
        resample = EEGPTResample(num_samples=1024)
        eeg = resample(torch.randn(2, 19, 800))

    Args:
        num_samples (int): Output samples per window. (default: :obj:`1024`)
    '''
    def __init__(self, num_samples=1024):
        super().__init__()
        _positive_int('num_samples', num_samples)
        self.num_samples = num_samples

    def forward(self, x):
        r'''Resample real floating input shaped [..., samples].'''
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or any(size == 0 for size in x.shape):
            raise ValueError('Input must be a nonempty tensor with a final sample axis.')
        if not x.is_floating_point():
            raise ValueError('Input must use a real floating dtype.')
        source_samples = x.shape[-1]
        if source_samples == self.num_samples:
            return x
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.rfft(work, dim=-1)
        shorter = min(source_samples, self.num_samples)
        bins = shorter // 2 + 1
        spectrum = spectrum[..., :bins]
        if shorter % 2 == 0:
            # Split an input Nyquist bin when upsampling; combine its pair
            # when downsampling. irfft discards the target Nyquist imaginary part.
            scale = 0.5 if source_samples < self.num_samples else 2.0
            spectrum = torch.cat((spectrum[..., :-1], spectrum[..., -1:] * scale), dim=-1)
        spectrum = F.pad(spectrum, (0, self.num_samples // 2 + 1 - bins))
        result = torch.fft.irfft(spectrum, n=self.num_samples, dim=-1)
        return (result * (self.num_samples / source_samples)).to(dtype=x.dtype)


class EEGPTInputAdapter(nn.Module):
    r'''Learn channel-slot features and resample whole EEG windows for EEGPT.

    Accept [batch, in_channels, input_samples] and return
    [batch, out_channels, num_samples]. Apply Fourier resampling, a constrained
    pointwise channel projection, BatchNorm/GELU, a depthwise temporal
    convolution, BatchNorm/GELU and dropout. Constructors do not run dummy
    data through the network; normalization statistics start clean.

    The output channels are learned feature slots, not reconstructed physical
    electrodes. When composing with EEGPT, declare the encoder's slot IDs
    explicitly and match their number/order to this output. The source channel
    order must remain consistent between training and inference. This module
    does not identify missing electrodes or infer montage, sampling rate,
    physical units or window duration.

    Resampling preserves the whole input window's duration. For example,
    800 samples at 200 Hz become 1024 samples at 256 Hz for the same four
    seconds. It assumes periodic windows, as documented by EEGPTResample.
    Temporal kernel lengths refer to the resampled axis. Training BatchNorm
    uses training windows only; call eval() for held-out subjects.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT, EEGPTClassifier, EEGPTInputAdapter, EEGPTAdaptedModel

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        adapter = EEGPTInputAdapter(in_channels=19, out_channels=3, num_samples=1024)
        model = EEGPTAdaptedModel(adapter, EEGPTClassifier(encoder, freeze_encoder=True))
        logits = model(torch.randn(2, 19, 800))

    The encoder's channel IDs in this example label learned slots. They do
    not assert that the adapter recovered actual C3/Cz/C4 sensor waveforms.
    Freezing the encoder retains input gradients, so the adapter and task
    head can train through it. An encoder starts from random weights unless
    a complete checkpoint is loaded explicitly.

    Args:
        in_channels (int): Number of source channels in the declared input order.
        out_channels (int): Number of learned output slots.
        num_samples (int): Output samples per whole input window. (default: :obj:`1024`)
        temporal_kernel (int): Positive odd depthwise kernel length after resampling. (default: :obj:`15`)
        dropout (float): Output dropout probability. (default: :obj:`0.25`)
        max_norm (float, optional): Per-output channel-projection norm limit; None disables projection. (default: :obj:`1.0`)
    '''
    def __init__(self, in_channels, out_channels, num_samples=1024,
                 temporal_kernel=15, dropout=0.25, max_norm=1.0):
        super().__init__()
        for name, value in [('in_channels', in_channels), ('out_channels', out_channels),
                            ('num_samples', num_samples), ('temporal_kernel', temporal_kernel)]:
            _positive_int(name, value)
        if temporal_kernel % 2 == 0:
            raise ValueError('temporal_kernel must be odd to preserve the sample count.')
        _probability('dropout', dropout)
        if max_norm is not None and (isinstance(max_norm, bool) or
                not isinstance(max_norm, (int, float)) or not math.isfinite(max_norm) or max_norm <= 0):
            raise ValueError('max_norm must be finite and positive, or None.')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_samples = num_samples
        self.temporal_kernel = temporal_kernel
        self.resample = EEGPTResample(num_samples)
        projection = nn.Conv2d(in_channels, out_channels, 1) if max_norm is None else \
            Conv2dWithConstraint(in_channels, out_channels, 1, max_norm=max_norm)
        self.chan_conv = nn.Sequential(
            projection,
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        r'''Adapt ordered EEG [batch, in_channels, input_samples] into learned slots.'''
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError('Input must have shape [batch, in_channels, samples].')
        return self.chan_conv(self.resample(x).unsqueeze(2)).squeeze(2)


class EEGPTAdaptedModel(nn.Module):
    r'''Compose an input adapter with a dimension-checked EEGPT task model.

    Input has the adapter's declared source channel order and sample count.
    The adapter produces learned feature slots for the task model input.
    A task may explicitly map these slots into its internal encoder geometry. The task model retains its classification, regression or
    embedding output. Named child modules provide stable state paths, and
    the constructor does not run data or update normalization statistics.

    Calling train() trains the adapter and task head. A task model configured
    with freeze_encoder=True keeps its encoder frozen and in evaluation mode;
    input gradients still pass through it to the adapter. Call eval() before
    evaluating held-out subjects to freeze adapter normalization statistics.

    .. code-block:: python

        from torcheeg.models import EEGPT, EEGPTClassifier
        from torcheeg.models import EEGPTInputAdapter, EEGPTAdaptedModel

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        adapter = EEGPTInputAdapter(19, 3, num_samples=1024)
        model = EEGPTAdaptedModel(adapter, EEGPTClassifier(encoder, freeze_encoder=True))

    Output slots are learned features rather than recovered physical sensors.
    Their order and the encoder's declared token IDs must be intentional.

    Args:
        adapter (EEGPTInputAdapter): Configured source-to-slot input adapter.
        model (nn.Module): EEGPTClassifier, EEGPTRegression, EEGPTEmbedding or EEGPTFinetuneClassifier.

    .. automethod:: feature
    '''
    def __init__(self, adapter, model):
        super().__init__()
        if not isinstance(adapter, EEGPTInputAdapter) or not isinstance(model, _EEGPTDownstream):
            raise ValueError('Use an EEGPTInputAdapter and an EEGPT task model.')
        input_channels = getattr(model, 'input_channels', model.encoder.num_electrodes)
        if adapter.out_channels != input_channels or adapter.num_samples != model.encoder.chunk_size:
            raise ValueError('Adapter output slots must match task input channels, and samples must match the configured encoder window.')
        self.adapter = adapter
        self.model = model
        self.feature_dim = model.feature_dim

    def forward(self, x):
        return self.model(self.adapter(x))

    def feature(self, x, flatten=False):
        r'''Extract task-encoder summaries after adapting the source input.

        Args:
            x (torch.Tensor): Ordered source EEG [batch, in_channels, samples].
            flatten (bool): Flatten the encoder summaries to [batch, feature_dim]. (default: :obj:`False`)
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        return self.model.feature(self.adapter(x), flatten=flatten)
