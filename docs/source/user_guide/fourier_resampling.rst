Differentiable Fourier resampling
=================================

:class:`torcheeg.models.FourierResample` resizes the last axis of a real floating
PyTorch tensor. It has no learned parameters or buffers and supports input
gradients. Set ``num_samples`` explicitly::

    import torch
    from torcheeg.models import FourierResample

    # Preserve four seconds: 128 Hz -> 200 Hz.
    eeg = torch.randn(2, 19, 512, requires_grad=True)
    resized = FourierResample(num_samples=800)(eeg)
    resized.square().mean().backward()

The layer uses PyTorch FFT operations at runtime. Its convention matches
``scipy.signal.resample(x, num_samples, axis=-1, window=None, domain='time')``
for real, time-domain, unwindowed input. It corrects unpaired Nyquist bins
when the shorter length is even. It accepts nonempty tensors of any rank
with a final sample axis, including noncontiguous and expanded broadcast
views. Equal input/output lengths return the input object directly.

Input/output retain floating dtype and device. Float16 and bfloat16 FFT work
is promoted to float32 and converted back; float64 input keeps its precision.
Signal units and retained-frequency amplitudes are unchanged. The layer does
not infer sampling rate, patch duration, channel order or physical reference.
Set these in the data protocol. Fourier interpolation assumes periodic input;
endpoint discontinuities can cause ringing. The
`SciPy reference <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html>`_
describes that convention.

Whole windows and independent patches
--------------------------------------

For input ``[batch, electrodes, window_samples]``, the last axis is the whole
window. For input ``[batch, electrodes, patches, patch_samples]``, each patch
is resampled independently::

    patches = torch.randn(2, 19, 4, 128)
    # Four one-second patches: 128 Hz -> 200 Hz for each patch.
    resized_patches = FourierResample(200)(patches)

These layouts are deliberate choices. Per-patch processing treats each patch
as periodic; whole-window processing treats only the window as periodic.
They generally give different results at patch boundaries. Do not flatten or
split an axis silently when adapting an existing model. The layer performs
no learned channel mixing, signal normalization or electrode recovery.

Compatibility and restored state
---------------------------------

:class:`torcheeg.models.EEGPTResample` retains its existing class name,
module path, default output length of 1024, constructor settings and behavior.
It is a compatibility subclass of the generic layer. Existing EEGPT input
adapters still instantiate ``EEGPTResample`` and preserve their state paths.
The standalone generic layer requires an explicit output length.

A state dictionary is empty because the layer has no parameters or buffers.
Save the ``num_samples`` constructor setting with a larger model's state;
recreate the same layer configuration before strict state restoration.
The output sample count cannot be recovered from an empty state dictionary.

Run and verify
---------------

:doc:`../auto_examples/examples_fourier_resample` runs a fixed synthetic
signal example with SciPy comparison, input gradients, the legacy interface,
independent-patch Nyquist behavior and a whole-window versus patchwise ramp
comparison. These are numerical checks, not EEG benchmark results.

Functional tests also check the independent SciPy linear operator and its
adjoint, odd/even lengths, broadcast-gradient accumulation, patch isolation,
low precision, copies, state restoration, random-state preservation, and
strict export/save/load of both public classes. The existing EEGPT input
adapter and composition tests cover the retained wrapper inside task models.
