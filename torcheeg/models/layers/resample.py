import torch
from torch import nn
from torch.nn import functional as F


# The module-class marker is absent in the tested PyTorch 2.4 runtime.
_HAS_STATIC_MODULES = tuple(int(part) for part in torch.__version__.split('.')[:2]) >= (2, 6)


class FourierResample(nn.Module):
    r'''Differentiable Fourier resampling along the final sample axis.

    Resize the one-sided spectrum and correct the unpaired Nyquist bin when
    the shorter length is even. This follows the real, time-domain,
    unwindowed Fourier-resampling convention of scipy.signal.resample.
    The runtime implementation uses PyTorch only. It treats each input
    window as periodic; discontinuities at its endpoints can cause ringing.

    Compiled output length remains guarded constructor configuration. FFT
    kernels in the tested runtimes require concrete input dimensions; use
    torch.compile(..., dynamic=False) for fixed-window workflows whose shapes
    change between compiled instances. PyTorch 2.4 also requires this option
    for changing constructor lengths because its module static marker is not
    available. Symbolic FFT inputs are not supported.

    Input and output use the same floating dtype and device. Float16
    and bfloat16 spectra are computed in float32, then converted back.
    Equal lengths return the input directly. No signal units or amplitude
    normalization are changed. The number of samples alone does not declare
    a sampling rate: the caller must preserve and state the window duration.

    - Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

    .. code-block:: python

        import torch
        from torcheeg.models import FourierResample

        # Four seconds at 200 Hz -> four seconds at 256 Hz.
        resample = FourierResample(num_samples=1024)
        eeg = resample(torch.randn(2, 19, 800))

    Args:
        num_samples (int): Required output samples per window.
    '''

    def __init__(self, num_samples):
        super().__init__()
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError('num_samples must be a positive integer.')
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
        # MKL FFT rejects zero-stride expanded batch axes; packing preserves
        # their values and broadcast gradient accumulation.
        spectrum = torch.fft.rfft(work.contiguous(), dim=-1)
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



# FFT output length is constructor configuration, not a symbolic input size.
# Specialize each instance's integer attributes while retaining value guards.
if _HAS_STATIC_MODULES:
    torch._dynamo.mark_static(FourierResample)
