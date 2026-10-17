EEGPT input adapters
====================

The direct EEGPT encoder uses explicit channel identities and a fixed sample
count. An input adapter is a separate trainable component for learning
channel-slot features and converting the whole window's sample count.

Declare the source and target contracts
----------------------------------------

Use :class:`torcheeg.models.EEGPTResample` for differentiable Fourier
resampling. It follows the real, unwindowed
`SciPy resample convention <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html>`_,
including even-length Nyquist-bin corrections. Input windows are treated as
periodic; endpoint discontinuities can ring. This module does not infer
sampling rate or change window duration.

For example, 800 samples at 200 Hz and 1024 samples at 256 Hz both describe
four seconds. Resampling 800 samples to 1024 has that rate interpretation
only when the caller preserves the four-second window duration. Amplitude
scaling and odd/even frequency-bin handling are checked independently.

:class:`torcheeg.models.EEGPTInputAdapter` then applies learned channel
projection, BatchNorm/GELU, depthwise temporal convolution, BatchNorm/GELU
and dropout. Input source channels must retain their declared order.
Output channels are learned feature slots. They are not reconstructed
physical electrodes, and the model does not infer a missing montage.

Use the named, checked task composition
----------------------------------------

.. code-block:: python

   import torch
   from torcheeg.models import EEGPT, EEGPTClassifier
   from torcheeg.models import EEGPTInputAdapter, EEGPTAdaptedModel

   encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
   task = EEGPTClassifier(encoder, freeze_encoder=True)
   adapter = EEGPTInputAdapter(in_channels=19, out_channels=3,
                               num_samples=1024)
   model = EEGPTAdaptedModel(adapter, task)
   logits = model(torch.randn(2, 19, 800))

Here C3/CZ/C4 identify the encoder's three learned slots. They do not claim
that the adapter recovered those three sensor waveforms from the source.
The encoder starts from random weights unless a complete checkpoint is
loaded explicitly. The adapter and task head need training in either case.

:class:`torcheeg.models.EEGPTAdaptedModel` checks the adapter output geometry
against the task encoder. It supports EEGPT classification, regression and
embedding task wrappers. Named child modules provide stable saved-state
paths. The constructor runs no dummy forward and leaves BatchNorm statistics
at their initial values.

Calling ``model.train()`` trains adapter normalization and the task head.
With ``freeze_encoder=True``, the task keeps its encoder frozen and in
evaluation mode, while retaining input gradients for adapter training.
Calling ``model.eval()`` also freezes the adapter's normalization statistics
for held-out subjects. Data-dependent normalization statistics must not be
fitted on the test subject.

Restore the dedicated example
-----------------------------

:doc:`../auto_examples/examples_eegmat_eegpt_input_adapter` reads 19 ordered
EEGMat channels, uses 200 Hz four-second source windows, and predicts
background versus arithmetic with a held-out subject. It preserves source
channel names separately from learned-slot names and records both sampling
rates and the window duration. The small random encoder verifies the
training path; the result is not pretrained EEGPT validation.

.. code-block:: python

   import torch
   from torcheeg.models import EEGPT, EEGPTClassifier
   from torcheeg.models import EEGPTInputAdapter, EEGPTAdaptedModel

   checkpoint = torch.load('eegmat-eegpt-input-adapter.pt',
                           map_location='cpu', weights_only=True)
   model = EEGPTAdaptedModel(
       EEGPTInputAdapter(**checkpoint['adapter_config']),
       EEGPTClassifier(EEGPT(**checkpoint['encoder_config']),
                       **checkpoint['head_config']))
   model.load_state_dict(checkpoint['state_dict'], strict=True)
   model.eval()

The checkpoint contains model tensors and configuration. It does not contain
source EDF recordings. Use the recorded source channel order, preprocessing,
window duration and rate when applying the restored model.
