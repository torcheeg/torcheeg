"""
Resample whole windows and independent patches
==============================================

FourierResample is a differentiable PyTorch layer for real floating signals
with shape [..., samples]. This example uses synthetic signals, not EEG
recordings or benchmark scores. It runs without a download or training job.

Resample a four-second 10 Hz signal from 128 Hz (512 samples) to 200 Hz
(800 samples), keeping duration and amplitude explicit. Compare with SciPy,
backpropagate into the input, and verify the existing EEGPTResample API.

For a tensor [batch, channels, patches, patch_samples], the same layer acts
on each patch independently. Each patch is treated as periodic. That differs
from resampling a whole window, whose periodic boundaries occur only at
its own endpoints. The ramp comparison below makes that distinction visible
numerically. Use the layout required by the model and state the patch duration.

Nyquist bins need special handling when an even shorter length changes.
The final check upsamples an 8-sample Nyquist oscillation independently in
four patches; its amplitude stays one. This is not a learned interpolation
or a recovery of missing physical samples. The layer does not infer rate,
reference, units, channel order or window duration.

Run::

    python examples/examples_fourier_resample.py
"""
import json

import numpy as np
from scipy.signal import resample
import torch

from torcheeg.models import FourierResample, EEGPTResample


def main():
    source_time = torch.arange(512, dtype=torch.float64) / 128
    source = torch.sin(2 * torch.pi * 10 * source_time).repeat(2, 3, 1).requires_grad_()
    result = FourierResample(800)(source)
    expected = torch.sin(2 * torch.pi * 10 * torch.arange(800, dtype=torch.float64) / 200).expand_as(result)
    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-11)
    scipy_result = torch.tensor(resample(source.detach().numpy(), 800, axis=-1))
    torch.testing.assert_close(result, scipy_result, rtol=1e-12, atol=1e-12)
    legacy_result = EEGPTResample(800)(source)
    torch.testing.assert_close(result, legacy_result, rtol=0, atol=0)
    result.square().mean().backward()
    assert torch.isfinite(source.grad).all() and source.grad.norm() > 0

    ramp = torch.linspace(-1, 1, 512, dtype=torch.float64).reshape(1, 1, 512)
    whole = FourierResample(800)(ramp)
    patched = FourierResample(200)(ramp.reshape(1, 1, 4, 128)).reshape(1, 1, 800)
    difference = (whole - patched).abs().max().item()
    assert difference > .01

    nyquist = torch.cos(torch.pi * torch.arange(8, dtype=torch.float64)).expand(2, 3, 4, -1)
    resized = FourierResample(16)(nyquist)
    expected_nyquist = torch.cos(2 * torch.pi * 4 * torch.arange(16, dtype=torch.float64) / 16).expand_as(resized)
    torch.testing.assert_close(resized, expected_nyquist, rtol=1e-12, atol=1e-12)
    report = dict(source_rate_hz=128, target_rate_hz=200, window_seconds=4,
        source_shape=list(source.shape), target_shape=list(result.shape),
        scipy_max_error=(result.detach() - scipy_result).abs().max().item(),
        legacy_max_error=(result.detach() - legacy_result.detach()).abs().max().item(),
        input_gradient_norm=source.grad.norm().item(),
        whole_vs_patch_ramp_difference=difference,
        patch_nyquist_max_error=(resized - expected_nyquist).abs().max().item())
    print(json.dumps(report))


if __name__ == '__main__':
    main()
