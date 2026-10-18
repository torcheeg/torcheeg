LaBraM encoder
===============

The existing :class:`torcheeg.models.LaBraM` classification interface and its
base/large/huge constructors are retained. LaBraMEncoder is a separate
interface for complete native encoder state and explicit feature contracts.

:class:`torcheeg.models.LaBraMEncoder` adapts the encoder from
`Jiang et al., ICLR 2024 <https://openreview.net/forum?id=QzTpTRVtrP>`_.
The implementation follows the author's MIT-licensed
`repository <https://github.com/935963004/LaBraM>`_ at revision
``c431221e6cfd23dbfa9950e0180682fb322b0548``. It retains all 221 native
base encoder tensors and their names without a timm runtime dependency.
This encoder does not include the spectrum tokenizer, masked-code predictor,
contrastive projection, optimizer or trained downstream heads.

Input, channels and physical time
----------------------------------

Supply real floating EEG as ``[batch, electrodes, samples]`` or explicit
``[batch, electrodes, patches, patch_size]`` patches. Native signals use
200 Hz and one-second, 200-sample patches. The author's fine-tuning code
uses microvolts divided by 100. Filtering, resampling, reference selection
and signal units belong to the caller's data pipeline. The encoder does
not infer or change these settings.

The temporal convolution has three Conv2d/GroupNorm/GELU stages. It flattens
time within a patch before convolution channels, matching the native tensor
layout. Its output width is ``ceil(patch_size / 8) * conv_channels`` and
must equal hid_channels. Conv channels must be divisible by four for the
native GroupNorm. Incomplete patches and inconsistent widths are errors.

Resolve electrode slots in source order::

    indices = LaBraMEncoder.channel_indices(['Fp1', 'C3', 'C4'])

Names ignore case and outer whitespace. Duplicate or unknown names and
unrecorded reference suffixes raise an error. Native bipolar identities
have separate slots: ``FP1-F7`` is not relabelled ``FP1``. The official
catalog lists 136 names, but the base checkpoint has only 129 table rows:
CLS at zero and 128 electrodes. The last eight catalog derivations, from
FP1-F3 through P4-O2, have no native slot and are explicitly rejected.
Integer indices use slots 1..128 and may be shaped ``[electrodes]``,
``[1, electrodes]`` or ``[batch, electrodes]``. CLS is inserted internally.

The learned temporal table has 16 patches by default. Shorter windows use
its prefix. Longer windows raise an error unless ``interpolate_time=True``
explicitly enables linear interpolation with align_corners=False. This
extension retains the original learned table rather than replacing its
shape or pretending that new positions were pretrained. A different
max_patches constructor value changes the learned tensor shape and must
match the supplied checkpoint.

Features and normalization
---------------------------

Tokens use electrode-major, temporal-minor order. ``forward_all`` returns
``[batch, 1 + electrodes * patches, hid_channels]``, with CLS first.
``forward_features`` excludes CLS and returns
``[batch, electrodes, patches, hid_channels]``. Both methods default to
``normalize=True``, applying the learned native final norm separately to
each token. False exposes raw final-block tokens.

Default ``model(eeg, indices)`` applies that learned norm AFTER mean pooling
the raw patch tokens. ``pooling='cls'`` selects the normalized CLS token.
Normalization after averaging differs from averaging normalized tokens.
The native pretraining tensor names remain norm.weight/norm.bias.

The author's fine-tuning model names its mean-pooling norm fc_norm, which
the released student norm does not automatically restore through a permissive
state load. TorchEEG explicitly reuses the learned native norm for pooling.
It does not claim that this choice reproduces a freshly initialized native
fine-tuning norm or a trained task head. Task-specific wrappers are separate
from this encoder.

Model variants
---------------

.. list-table:: Native raw-EEG encoder dimensions
   :header-rows: 1

   * - Variant
     - hid_channels
     - conv_channels
     - mlp_channels
     - depth
     - num_heads
   * - Base
     - 200
     - 8
     - 800
     - 12
     - 10
   * - Large
     - 400
     - 16
     - 1600
     - 24
     - 16
   * - Huge
     - 800
     - 32
     - 3200
     - 48
     - 16

All use patch_size=200. Base is the default. Dimensions alone do not prove
that an author's checkpoint or benchmark was reproduced. Only the official
base checkpoint has been checked numerically here.

Complete local checkpoint loading
-----------------------------------

``from_pretrained`` accepts tensor dictionaries, model or state_dict
containers. The prefix is explicit and containers must be unambiguous.
Every encoder tensor, shape, floating type and finite value is required.
There is no silent partial restoration or norm/fc_norm rename.

The author's tracked ``checkpoints/labram-base.pth`` contains a model with
student-prefixed tensors, pretraining heads and training metadata. Use::

    model = LaBraMEncoder.from_pretrained(path, prefix='student.',
        exclude_pretraining=True, allow_training_metadata=True)

The exclusion removes only the selected student's mask_token and lm_head.
Other prefixes are not selected. The complete 221-tensor base encoder is
restored, including spatial position embeddings, query/key norms and final
norm. No weights or preprocessing are downloaded or inferred.

Training metadata opt-in uses weights_only=True and a scoped safe_globals
context limited to argparse.Namespace, the known NumPy scalar constructor,
NumPy dtype and float64 dtype class. It does not execute arbitrary pickle
objects or persistently expand the global allowlist. This opt-in requires
the public safe_globals context in tested PyTorch 2.6/2.14; on 2.4 convert
the verified file to a tensor-only state using a supported runtime first::

    torch.save(model.state_dict(), 'labram-base-encoder.pt')

Load that complete tensor-only file without prefix or metadata opt-in.
Verify the source and hash before enabling training metadata loading.


The tracked base artifact has SHA256
``7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c``.
Its origin is the pinned author repository, not an unrelated model mirror.

Teaching workflow
------------------

:doc:`/auto_examples/examples_eegmat_labram` trains a linear probe on public
EEGMat background/arithmetic labels, holding out a whole subject. Complete
recordings are resampled to 200 Hz, bandpass-filtered .1--75 Hz and notch
filtered at 50 Hz with MNE FIR. All 19 source channels and their reference
are retained. Microvolts are divided by 100 without per-window z-scoring.

The encoder is frozen in evaluation mode, while a new dropout/linear head
trains for fixed epochs. It uses the learned native norm after patch-token
mean pooling, explicitly distinct from a newly initialized fc_norm. The
artifact contains every encoder/head tensor, both constructor configurations,
preprocessing/channel/partition settings and all held-out outputs/targets.
Rebuild those configurations and strictly restore complete tensor state.
No optimizer state is saved or resumed.

The default encoder is small and random; an explicitly verified local base
checkpoint is optional. The example downloads nothing. A teaching run
verifies training and restoration, not published benchmark accuracy or a
pretrained behavioral claim. Test labels do not select epochs or parameters.
