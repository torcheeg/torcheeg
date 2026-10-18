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


Existing classifier compatibility
----------------------------------

:class:`torcheeg.models.LaBraM` retains its classification API, constructor
parameters and base/large/huge factories. It is separate from LaBraMEncoder.
Setting use_abs_pos_emb=False now disables the spatial lookup while retaining
temporal positions, all token-return modes and the existing pooling/head
behaviour. It does not replace the spatial table with a random substitute.

Its :meth:`~torcheeg.models.LaBraM.load_pretrained` method accepts local
raw tensor state or one model, module or state_dict container. None of these
flat states is silently filtered into an empty dictionary. With prefix=None,
it selects student. when present and otherwise retains raw keys; an explicit
prefix selects and strips exactly that prefix. Empty, wholly unrelated or
ambiguous selections are errors. For example::

    model = LaBraM.base_patch200_200(num_classes=4)
    result = model.load_pretrained('complete-classifier.pt', strict=True)

strict=True requires complete matching classifier state. The default
strict=False remains a transfer-loading policy: every missing and unexpected
key is reported in a warning and returned as load_state_dict's missing_keys
and unexpected_keys. No compatible tensor is copied until all matching
tensor shapes/types and finite values have been checked. Invalid shape or
nonfinite input fails before any partial mutation. exclude_head=True
explicitly leaves head.* fresh, for transfer across class counts; it does
not silently ignore an incompatible head by default.

A complete native encoder's norm and the mean-pooled classifier's fc_norm
have distinct roles. They are not automatically renamed. Native transfer
therefore reports the missing classifier head/fc_norm and unexpected native
norm/pretraining tensors. Check the returned report. Complete native encoder
restoration with explicit token contracts remains LaBraMEncoder's role.

Both loaders always use weights_only=True across tested runtime defaults.
They share the existing known-author metadata context. A verified author
training file needs explicit allow_training_metadata=True to admit its
Namespace and recognized NumPy scalar metadata, with no unsafe fallback or
persistent allowlist additions. Runtimes lacking the public safe_globals
context require verified conversion to tensor-only state first. Executable
pickle metadata is rejected even with the known-metadata option. Loading
performs no download, optimizer restoration or parameter-mode changes.


The legacy electrode argument keeps its existing standard_1020 spellings
(e.g. FP1 rather than Fp1). Its query/key normalization now uses a two-dimensional
per-head view to avoid backward-fusion layout conflicts in tested compiler
runtimes. Output/token/pooling contracts and native parameter paths remain
unchanged. Full-graph training checks cover spatial positions enabled or
disabled and mean or CLS pooling, with input and all parameter gradients
compared against eager execution.


The legacy catalog lists 136 electrode/derivation names, while its native
spatial table has 128 electrode slots plus CLS. With spatial positions enabled,
provide explicit electrodes within those learned slots. The first eight
bipolar names retain slots 121--128; the last eight catalog entries have no
learned spatial vectors and raise a clear error. The implicit full catalog
also raises before a tensor shape failure. Disabling spatial positions
removes this lookup constraint without inventing or remapping positions.
