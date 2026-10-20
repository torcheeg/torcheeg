BIOT pretrained encoders
========================

:class:`torcheeg.models.BIOT` implements the native 256-wide, eight-head,
four-block encoder from `BIOT (Yang et al., NeurIPS 2023)
<https://github.com/ycq091044/BIOT>`_. :class:`torcheeg.models.BIOTClassifier`
adds the author's ELU/linear classification head. Both frozen probing and
complete downstream fine-tuning use ordinary PyTorch optimizers. Task-head
weights are newly initialized; encoder weights alone do not contain a
pretrained task classifier. The self-supervised pretraining prediction head,
perturbation sampler and pretraining loss are outside this encoder API.

Architecture and inputs
-----------------------

Supply real floating ``[batch, electrodes, samples]`` signals. Official
settings require 200 Hz, ``n_fft=200``, ``hop_length=100``. Rectangular-window
STFT uses no centering, magnitude without log or scaling, then a linear
frequency projection. Add channel tokens and sinusoidal positions restarting
at zero for each channel. Concatenate channel-major tokens, apply native
noncausal linear attention and mean-pool to ``[batch, 256]``. Linear attention
factorizes separately normalized queries/keys; ordinary quadratic softmax
attention is a different architecture. No optional attention package is needed.

The source's global attention has no actual 1024-token restriction; each
channel is limited to the available 1000 positional entries. Custom widths
and FFT settings construct random-model architectures, not official variants.
The model performs no unit conversion, resampling, montage construction,
normalization or artifact removal. These steps belong to the data protocol.

All three official variants
---------------------------

Files are from author revision ``d138e32634e52ae9fa6ec98ac9c4087b14ca869a``.
Download from its ``pretrained-models`` directory explicitly and verify SHA256:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Variant
     - Official filename
     - SHA256
   * - ``prest-16ch``
     - ``EEG-PREST-16-channels.ckpt``
     - ``40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70``
   * - ``shhs-prest-18ch``
     - ``EEG-SHHS+PREST-18-channels.ckpt``
     - ``391b53f77d5060e43746f3ffd5aae8107785f4fd122bf2f87221a1b5b2854a84``
   * - ``six-datasets-18ch``
     - ``EEG-six-datasets-18-channels.ckpt``
     - ``78ff15a1782f194286a97b1abe68b2bf100a39803325e2917337d0a77a228542``

The first 16 token IDs, in order, are::

    FP1-F7, F7-T7, T7-P7, P7-O1,
    FP2-F8, F8-T8, T8-P8, P8-O2,
    FP1-F3, F3-C3, C3-P3, P3-O1,
    FP2-F4, F4-C4, C4-P4, P4-O2

The 18-token variants append ``C3-A2, C4-A1``. ``channel_ids`` selects native
token IDs corresponding to the actual input channels; it cannot repair a
wrong montage. Without IDs, the first input channel maps to token zero.
Select IDs explicitly for reordered/subset input. Do not label an arbitrary
19-electrode unipolar tensor as the pretrained bipolar montage.

Strict loading and classification
---------------------------------

.. code-block:: python

    from torcheeg.models import BIOT, BIOTClassifier

    backbone = BIOT.from_pretrained(
        './EEG-PREST-16-channels.ckpt', variant='prest-16ch')
    model = BIOTClassifier(backbone, num_classes=2, freeze_backbone=True)
    model.train()  # frozen backbone remains in eval; only head trains
    logits = model(eeg)  # [batch, 16, samples] -> [batch, 2]

Set ``freeze_backbone=False`` to optimize the full model. For inference call
``eval()`` and disable gradients. Feature extraction uses ``backbone(eeg)``
or ``backbone.forward_features(eeg)``. Constructor names follow TorchEEG:
``num_electrodes``, ``hid_channels`` and ``num_heads`` correspond to native
``n_channels``, ``emb_size`` and ``heads``.

``from_pretrained`` uses ``weights_only=True`` with no unsafe fallback. It
accepts a tensor dictionary or a ``state_dict`` container and an explicit
optional prefix. It requires all 57 keys, exact shapes/dtypes, finite values,
canonical channel indices and sinusoidal positions. The official position
table differs across CPU/GPU trigonometric kernels by at most 3.06e-5; a
4e-5 absolute tolerance validates that fixed table while retaining the
supplied tensor. Weights are never silently skipped. The two 18-channel
variants share shapes: verify the checksum to establish which weights you use.

Standalone real-data workflow
-----------------------------

``examples/examples_eegmat_biot.py`` requires a full official checkpoint,
constructs all 16 bipolar derivations, converts microvolts to volts, and
normalizes each four-second channel/window by its 95th percentile absolute
amplitude plus 1e-8 volts. This follows the author's loader scaling with an
explicit source-unit conversion. No data-dependent population statistic is
fit and no extra filter is applied. The last sorted subject is held out;
all other subjects train the frozen encoder's classification head. Both
18-token checkpoints intentionally omit their two unavailable ear-reference
channels. No zero channels are fabricated. Use the example's CLI help for
paths. Saved full state, JSON metadata and held-out predictions are loaded
strictly and reproduced before reporting success.

Source and validation
---------------------

BIOT and the used noncausal global attention subset from
``linear-attention-transformer==0.19.1`` are MIT licensed; notices are retained
in ``LICENSES/BIOT.txt``. The architecture regression fixture independently
uses the unmodified author implementation and records both source hashes.
It checks full-width outputs and every trainable parameter/input gradient
in evaluation, training with dropout, zero-input and offset-channel cases.
The three official checkpoints were separately checked against that source;
all output and input-gradient maximum errors were zero on PyTorch 2.6 CPU,
with every parameter gradient also matching. These checks establish native
compatibility, not published benchmark performance. Real EEG teaching runs,
GitHub CI and Read the Docs cloud builds are separate verification steps.
