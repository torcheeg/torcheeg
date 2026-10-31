EEGWaveNet with the complete external author network
====================================================

:class:`torcheeg.models.EEGWaveNet` retains the complete supervised network
from `IoBT-VISTEC/EEGWaveNet <https://github.com/IoBT-VISTEC/EEGWaveNet>`_.
It contains six learned depthwise temporal reductions, five independent
two-convolution channel-pooling branches with BN/LeakyReLU, temporal means,
160-feature concatenation and the full three-layer classifier, including its
intermediate sigmoid and final log-softmax. It is not a pretrained foundation
model and does not use an inferred wavelet transform.

Supply a trusted original file
------------------------------

The inspected author repository does not state a license. TorchEEG provides
an interface and does not bundle or download the implementation or weights.
Obtain ``architecture.py`` at revision
``3b1909873f6cde2d8b5a916327791cf9a59f975c`` separately. The factory verifies
SHA256 ``6922fecd52bfd240dca4ce7475ea94657fc750b2e05f65e2eda52cfa09caf408``
before explicitly executing that local file. It requires only PyTorch and
registers the complete verified module for normal PyTorch resolution.
Importing TorchEEG does not execute the author file.

.. code-block:: python

   import torch
   from torcheeg.models import EEGWaveNet

   model = EEGWaveNet.from_source(
       './author/architecture.py', num_electrodes=22,
       chunk_size=1000, num_classes=4)
   eeg = torch.randn(2, 22, 1000)
   log_probabilities = model(eeg)
   loss = torch.nn.functional.nll_loss(
       log_probabilities, torch.tensor([0, 3]))
   loss.backward()

An already constructed complete original ``architecture.Net`` can instead
be supplied as ``EEGWaveNet(backend, chunk_size=1000)``. The full model remains
registered in ``model.backend``; no branch or readout is replaced.

Input and native output
-----------------------

Provide floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``. Sampling rate, ordered channel names, units
and preprocessing are supplied by the caller. No filtering or normalization
is implicit. Public defaults use 22 rows, 1000 samples and four classes; the
author's README demonstrates 21 rows, 1024 samples and two classes. Both are
supported by the unchanged complete network.

The deepest temporal scale has ``floor(chunk_size / 64)`` points. Two valid
four-point convolutions then require at least seven points, so the minimum
input length is 448. Shorter inputs are rejected before source execution.
Incomplete pairs at each stride-two reduction are discarded by native
convolution; no padding or interpolation is added. BN requires enough values
per channel in training, as in the original model.

The native output is **log probabilities**, not raw logits or probabilities.
Use ``NLLLoss`` directly; use ``log_probabilities.exp()`` only when normalized
probabilities are required for reporting. Passing these normalized scores to
the standard classifier's cross entropy is mathematically equivalent, but
adds a redundant log-softmax. The independent example retains native NLL.
The author's extra ``training`` forward argument is unused; ``train()`` and
``eval()`` correctly control the retained BN modules.

Keep the complete checkpoint and protocol
-----------------------------------------

``model.state_dict()`` preserves all original states under ``backend.``.
Use ``native_state_dict()`` and ``load_native_state_dict(state, strict=True)``
for the complete author format without that prefix. Reconstruct from the
separately supplied source and load tensors strictly. Save model settings,
source hash/revision, physical input metadata, labels and split information;
do not pickle the external author Python object.

The independent
:doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_eegwavenet>` uses
22 EEG channels at 250 Hz in provider order and microvolts. It filters the
recording at 0.5--40 Hz, selects only the first four-second cue window per
trial, skips marked artifacts, normalizes each window per channel and holds
out the last sorted subject. Fixed epochs do not select checkpoints on test
labels. The atomic export includes complete model/optimizer/random states,
test log probabilities and source/protocol metadata and refuses overwrite.
CLI training resume is not implemented.

This is a motor-imagery transfer teaching workflow. The original paper studied
seizure detection on CHB-MIT and TUSZ; the example does not reproduce those
data protocols or scores. The fixed EEG-Arena backbone also adds padding to
its channel-pooling convolutions and uses its benchmark head. That is a
different variant: this interface preserves the full original valid
convolutions and classifier, without claiming Arena numerical equivalence.

Validation scope
-----------------

Native tests use the complete pinned author file. They check default
initialization/RNG, every state, train/eval outputs, normalized probabilities,
NLL, input/all-parameter gradients, BN buffers, AdamW updates, minimum and
alternative geometry, strict state loading, next-step optimizer resume,
sample isolation and eager full-graph compile. The dedicated CI supplies
the separate source by hash. Source-dependent tests explicitly skip if it
is unavailable; no small substitute model is used. Real-data functional
checks do not establish published or clinical performance. GPU, AMP,
optimized compilation and distributed execution are not claimed.
