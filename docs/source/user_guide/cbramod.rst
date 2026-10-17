CBraMod backbone and local weights
==================================

:class:`torcheeg.models.CBraMod` combines temporal convolution and normalized
real-FFT magnitude, adds a depthwise positional convolution, and processes
separate spatial/temporal feature halves with criss-cross attention.
Input shape is ``[batch, electrodes, patches, patch_size]``. Channel order,
reference, sample rate, signal units and physical patch duration are part
of the caller's data protocol.

Reconstruction and features
----------------------------

``model(x, mask=None)`` returns the native reconstruction projection with
shape ``[batch, electrodes, patches, out_features]``. This matches the official
backbone rather than silently discarding its projection. It does not define
a complete pretraining loss or masking schedule.

``model.forward_features(x, mask=None)`` returns features before projection,
with final width ``hid_channels``. ``model.feature(x, flatten=True)`` flattens
those features from axis one; the default leaves their four axes intact.
A downstream head must explicitly select pooling and output semantics::

    import torch
    from torcheeg.models import CBraMod

    model = CBraMod()
    eeg = torch.randn(2, 19, 4, 200)
    reconstructed = model(eeg)
    features = model.forward_features(eeg)
    pooled = features.mean(dim=(1, 2))

Optional masks have shape ``[batch, electrodes, patches]``. True/1 replaces
that complete patch with the native fixed mask encoding before temporal
and spectral embedding. Boolean masks support graph capture without
value-dependent binary validation; binary integer and floating masks are
also validated for eager execution. Complex magnitude uses ``abs`` so zero
signals have finite gradients. Low-precision FFT input is promoted to
float32; float64 inputs retain their precision.

Canonical configuration
-------------------------

The official checkpoint uses one-second, 200-sample patches at 200 Hz,
feature width 200, feed-forward width 800, 12 blocks and eight heads.
The positional convolution has spatial/temporal kernel ``(19, 7)``.
The model does not infer duration, reorder electrodes or resample signals.

Constructor names follow TorchEEG model conventions:

======================= =========================
TorchEEG argument       Official source argument
======================= =========================
``patch_size``          ``in_dim``
``out_features``        ``out_dim``
``hid_channels``        ``d_model``
``mlp_channels``        ``dim_feedforward``
``depth``               ``n_layer``
``num_heads``           ``nhead``
======================= =========================

The source's unused sequence-length constructor setting is omitted; the
backbone accepts explicit nonempty channel and patch axes. A changed patch
size or feature width is a separate random-model configuration, not the
canonical pretrained model. The convolution produces
``ceil(patch_size / 25)`` temporal steps. Its output width must divide
``hid_channels`` exactly; head counts must be even and divide that width.
Default convolution/group-normalization shapes preserve all native keys.

Safe complete component states
--------------------------------

The factory loads a local complete state with ``weights_only=True``::

    backbone = CBraMod.from_pretrained('pretrained_weights.pth')
    # A component inside a safely loadable state_dict checkpoint:
    backbone = CBraMod.from_pretrained('task.pt', prefix='backbone.')

It validates all selected names and shapes, floating dtypes and finite values,
including the native reconstruction projection. Missing or extra selected
keys are errors. Other components outside an explicit prefix are ignored.
Unsupported pickle objects anywhere in the file are rejected, with no unsafe
fallback. No weights are downloaded automatically. Constructor settings are
explicit: tensor shapes cannot establish signal units or preprocessing.

The `official repository <https://github.com/wjq-learning/CBraMod>`_ links the
`author's checkpoint <https://huggingface.co/weighting666/CBraMod>`_. The
validated file at model revision ``500543c7e30bda1b22bfd51a49301b238dee21fd``
is ``pretrained_weights.pth`` (19,775,842 bytes), with LFS SHA256
``0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178``.
The code carries MIT terms and the publisher's model card declares
Apache-2.0 for this weight repository. Retain the relevant notices when
using artifacts; TorchEEG packages code and its retained notice, not these
weight files.

A concrete frozen-backbone workflow
------------------------------------

:doc:`../auto_examples/examples_eegmat_cbramod` uses EEGMat recordings at
200 Hz, filters complete recordings at 0.3--75 Hz and 60 Hz notch, scales
microvolts by 1/100, and explicitly reorders all 19 channels into the official
pretraining order. The EEGMat source order differs from that order.
Four-second windows form four one-second patches. It trains a local linear
probe with the complete backbone frozen and its dropout in evaluation mode.
An optional complete local canonical checkpoint replaces the default small
random backbone.

The example saves full tensor state, constructor settings, source/target
channel order, filters, scale, checkpoint hash and all held-out logits/targets.
It preserves the source EEG reference. The linear probe is an explicit
teaching head; the full benchmark's task-specific multi-layer heads and
training schedule are separate. Its four-second task windows do not
reproduce pretraining trimming, 30-second segments or artifact rejection.
A one-epoch run establishes functionality, not a published benchmark score.

Independent evidence and provenance
------------------------------------

The implementation adapts the MIT-licensed
`official source at b9e9610 <https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a/models>`_.
The license and adaptation notice are retained in ``LICENSES/``. Independent
float64 fixtures from unmodified source check reconstruction, backbone
features, masking, zero signals, a changed reconstruction width, and all
input/parameter gradients. They include the canonical 12-block architecture.
A separate author-checkpoint comparison strictly loads all 211 native state
tensors (4,924,000 parameters) and compares outputs and gradients against
the unmodified official model. These numerical checks are distinct from
real downstream evaluation.
