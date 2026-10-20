BENDR complete pretrained backbone
==================================

:class:`torcheeg.models.BENDR` combines the complete convolutional encoder
and contextual transformer from `BENDR (Kostas et al., 2021)
<https://github.com/SPOClab-ca/BENDR>`_. The implementation is adapted from
BSD-licensed `DN3 <https://github.com/SPOClab-ca/dn3>`_, revision
``4d477fe42d3d8ce64f3b790585bfa5c7acb84848``. Notices are retained in
``LICENSES/BENDR.txt``. No DN3 installation is required.

Both official components
------------------------

The author's `v0.1-alpha release
<https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha>`_ provides:

* ``encoder.pt``: SHA256
  ``32c913b1778d8bdf7f08bbe8ef5e6bef2954087b482851816deefc4d0fcf84cb``.
* ``contextualizer.pt``: SHA256
  ``b8ee0c8c8403809d65114ae88485cdf841dd1860a88ebebb4cdee2bf26f91526``.

These weights use 20 input channels, encoder width 512, eight attention
heads, eight contextual blocks and feed-forward width 3076. The context
attention width is three times the encoder width (1536). Download the two
files explicitly and verify their source/checksums. They contain 24 and 76
native tensors respectively; neither component alone is the full backbone.

.. code-block:: python

    from torcheeg.models import BENDR, BENDRClassifier

    backbone = BENDR.from_pretrained('./encoder.pt', './contextualizer.pt')
    backbone.eval()
    representation = backbone(eeg)  # [batch, 20, samples] -> [batch, 512]
    sequence = backbone.forward_features(eeg)
    # [batch, 512, ceil(samples / 96) + 1], including start token at index zero
    model = BENDRClassifier(backbone, num_classes=2, freeze_backbone=True)

The six three-tap convolution blocks downsample by 3,2,2,2,2,2 and apply
channel dropout, GroupNorm and GELU. :class:`torcheeg.models.BENDREncoder`
returns only their sequence. :class:`torcheeg.models.BENDRContextualizer`
uses a grouped, weight-normalized relative-position convolution, conditioning,
a prepended constant start token, norm-free T-fixup transformer blocks and
output projection. The native unused final norm remains in the state for
complete compatibility. Forward selects the contextualized start token;
it does not accidentally select the first batch item.

The downstream head is newly initialized. With ``freeze_backbone=False``
the complete model can train; freezing also keeps backbone dropout disabled
when ``train()`` is called. Explicit boolean ``mask_t`` and ``mask_c`` masks
have shapes ``[batch, encoded_frames]`` and ``[batch, 512]``. Time replacement
precedes channel-zero masking; caller tensors are never modified. Automatic
random pretraining masks and a contrastive pretraining objective are outside
this API. Use ``eval()`` for deterministic features. Training layer drop
retains the source's random-draw order, including its draw at zero probability.

Input representation
--------------------

Official inputs use 256 Hz. The first 19 channels follow DN3 To1020 order::

    FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ,
    C4, T8, T5, P3, PZ, P4, T6, O1, O2

The twentieth channel is a scale auxiliary, not EEG. Native Deep1010 mapping
normalizes the EEG group together to [-1,1] over a complete window, including
zero-filled unmapped EEG rows when computing extrema, then resets missing
channels to zero. A known population maximum range can supply a relative
scale; its source/fitting population must be declared and must not use held-out
data in a standard holdout. Native ``max_scale=None`` sets the auxiliary to
zero. Missing-electrode mapping, aliases, physical units, reference, filters,
resampling and scale semantics are data protocol choices. The model does not
infer or apply them. Arbitrary 20-electrode raw EEG is not equivalent input.

Standalone EEGMat workflow
--------------------------

``examples/examples_eegmat_bendr.py`` runs the full pretrained model on actual
EEGMat recordings, resampled from 500 to 256 Hz by the reader. It reorders
all 19 available source channels, declares T3/T4 as T7/T8 aliases, applies
native-style global min/max normalization per four-second window, and appends
the explicitly unknown scale auxiliary zero. No scalp electrode is fabricated;
no additional filter, clipping or learned population statistic is applied.
Microvolt source units are documented before dimensionless normalization.
The last sorted subject is held out, all others train a linear head on the
frozen complete backbone, and a fixed epoch count avoids test-based selection.
This teaching protocol does not reproduce the original paper's full benchmark.

Both local official checkpoints are mandatory; the example downloads nothing
and has no random fallback. Saved complete state, constructor/preprocessing
JSON, source/weight hashes, subject IDs, held-out logits and labels support a
strict complete-model restoration check before success is reported.

Compatibility and validation
----------------------------

Loading uses ``weights_only=True`` and explicit CPU mapping, with no unsafe
fallback or silently omitted weights. Exact key sets, tensor shapes/dtypes and
finite values are required. Optional prefixes select complete components
explicitly. The old weight-normalization state layout (``weight_g/weight_v``,
native ``dim=2``) is retained. PyTorch currently warns that this public helper
is deprecated; replacing it requires separate numerical/checkpoint verification.
The implementation expresses norm-free context blocks directly, retaining the
native names and operations without private transformer helpers.

Independent regression fixtures check evaluation, controlled-dropout training,
zero signals, masking, outputs and every input/parameter gradient against
unchanged DN3 class bodies. Native random dropout is checked separately.
Full actual 512-wide/eight-layer author weights also match that source's
complete sequence and every gradient; output/input-gradient maximum errors
were zero on PyTorch2.6 CPU. Component tests, real-data execution, GitHub CI,
local documentation and Read the Docs cloud builds are separate checks.
