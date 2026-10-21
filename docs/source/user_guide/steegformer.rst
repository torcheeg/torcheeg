ST-EEGFormer
============

:class:`torcheeg.models.STEEGFormer` provides the encoder from
`Yang et al., ICLR 2026 <https://openreview.net/forum?id=5Xwm8e6vbh>`_.
The adaptation follows the author's MIT-licensed
`implementation <https://github.com/LiuyinYang1101/STEEGFormer>`_ at revision
``542ee17918c3c2c36ba1d4ea02bedff5eb149370``. It retains native encoder
tensor names without requiring timm or torchvision.
:class:`torcheeg.models.STEEGFormerPretrainer` separately provides the complete
masked autoencoder, including its native decoder. Task-specific pretrained
heads are not included.

Input and channel identities
-----------------------------

Supply real floating EEG shaped ``[batch, electrodes, samples]`` and an
explicit integer channel-index tensor shaped ``[electrodes]``,
``[1, electrodes]`` or ``[batch, electrodes]``. The model performs no
filtering, resampling, amplitude scaling or signal normalization.
Native pretraining uses 128 Hz signals and patches of 16 samples, or 125 ms.
Authors recommend windows up to six seconds. The native positional table
has 512 entries: CLS uses zero and patches use positions one through 511.
This architectural limit is distinct from the pretraining window length.

The sample count must be a positive multiple of patch_size and fit within
max_patches. Invalid windows raise an error rather than losing a partial
last patch. A temporal patch is projected separately for every electrode.
Patch tokens use temporal-major, channel-minor order.

Resolve names in the supplied source order::

    indices = STEEGFormer.channel_indices(['Fp1', 'Fp2', 'Cz'])

The canonical table has 145 learned slots and 142 named electrodes. Names
are matched exactly after trimming outer whitespace and ignoring case.
The remaining slots require explicit integer indices. Unknown names,
reference suffixes and bipolar derivations raise an error; they are not
silently relabelled as Cz or a single electrode. Select channel/reference
conversion in preprocessing and record it with the experiment.

Two feature contracts
----------------------

The native pretraining encoder applies a final LayerNorm. The author's
downstream global-pooling path omits that norm. They produce different
features even with identical encoder tensors.

``model(eeg, indices)`` returns the mean of normalized patch tokens,
excluding CLS, shaped ``[batch, hid_channels]``. This preserves the
original TorchEEG workspace's encoder representation.

``forward_all(eeg, indices, normalize=True)`` returns normalized tokens
including CLS, shaped ``[batch, 1 + patches * electrodes, hid_channels]``.
``forward_features(..., normalize=True)`` removes CLS and reshapes to
``[batch, patches, electrodes, hid_channels]``.

For the official downstream mean-pooling representation, use::

    features = model.forward_features(eeg, indices, normalize=False)
    pooled = features.mean(dim=(1, 2))

The final encoder norm remains in the complete state but receives no
gradient in this mode. Add an explicit task head to pooled features.
Do not compare results across these two modes without naming the choice.

Model sizes and state
----------------------

.. list-table:: Canonical encoder settings
   :header-rows: 1

   * - Variant
     - hid_channels
     - mlp_channels
     - depth
     - num_heads
   * - Small
     - 512
     - 2048
     - 8
     - 8
   * - Base
     - 768
     - 3072
     - 12
     - 12
   * - Large
     - 1024
     - 4096
     - 24
     - 16

All use patch_size=16, num_channel_embeddings=145 and max_patches=511.
Small is the default. Set all constructor dimensions explicitly when
loading another variant. Architecture settings alone do not establish
that a checkpoint or a published result was reproduced.

``from_pretrained`` accepts a local tensor dictionary, state_dict container
or native model container. It selects one explicit prefix and requires
every encoder tensor, exact shapes, floating types and finite values.
Missing/extra encoder keys and ambiguous containers are errors.
No weights are downloaded or preprocessing settings inferred.

The author's native small training file also contains optimizer state,
MAE decoder tensors and argparse.Namespace metadata. Explicitly setting
``exclude_decoder=True`` discards only recognized decoder components;
this restores the complete encoder, not the complete MAE. Setting
``allow_namespace=True`` permits only that standard-library metadata class
inside a safe loading context. Loading always uses weights_only=True,
without an unsafe fallback or a persistent global allowlist change.

That opt-in requires the public torch.serialization.safe_globals context,
available in the tested PyTorch 2.6 and 2.14 runtimes. PyTorch 2.4 has no
such public context. Convert a verified native file on a supported runtime,
then load its tensor-only encoder state on 2.4::

    model = STEEGFormer.from_pretrained(
        path, exclude_decoder=True, allow_namespace=True)
    torch.save(model.state_dict(), 'steegformer-small-encoder.pt')

Verify the source and hash before opting in to native metadata. The author
small release asset is ``checkpoint-300.pth`` (393972208 bytes), SHA256
``98d2848abdaea7a143d399c07f67f8782e07dd66e7c309ddb058c7158a8a8b02``.
No checkpoint is redistributed with TorchEEG.

Teaching workflow
------------------

:doc:`/auto_examples/examples_eegmat_steegformer` trains a linear probe on
the public EEGMat background/arithmetic task. It uses complete-recording
128 Hz resampling and .5--45 Hz FIR filtering, then per-channel, per-window
population z-scoring. Four-second windows contain 32 native temporal patches.
Source channel order and the source EEG reference are preserved.

The encoder remains frozen and in evaluation mode while the dropout/linear
head trains. The example explicitly uses the pre-norm downstream feature
contract. It saves complete tensor/configuration/channel/preprocessing/
partition state and every held-out output. A small random encoder is the
default; a verified local canonical small checkpoint is optional.

Subjects remain separate and epochs are fixed. A teaching run checks
training and safe restoration. It does not reproduce the author's benchmark
protocol or establish an EEG-Arena score.


Classification, regression and embeddings
------------------------------------------

:class:`torcheeg.models.STEEGFormerClassifier` returns unnormalized
``[batch, num_classes]`` logits. :class:`torcheeg.models.STEEGFormerRegression`
returns ``[batch, num_outputs]`` without an output activation. Both mean-pool
patch tokens over electrodes and time, then apply dropout and a linear head.
Default ``normalize_features=False`` matches the official downstream pre-norm
path; True explicitly selects the archived workspace's post-norm features.

:class:`torcheeg.models.STEEGFormerEmbedding` has no learned task head.
Default ``pooling='mean'`` returns ``[batch, hid_channels]``.
``pooling='none'`` returns a channel-major
``[batch, electrodes, patches, hid_channels]`` grid; ``pooling='flatten'``
flattens that grid from axis one. Each wrapper's ``feature`` method exposes
the same grid, with optional flattening. Its axis order differs from the
encoder's temporal-major grid.

Supply ordered ``channel_names``, source ``chunk_size`` and ``sampling_rate``.
The wrapper Fourier-resamples the entire window to 128 Hz. Physical duration
must map to an exact integer sample count and complete native patches;
invalid durations raise an error without truncating samples. Fourier
interpolation assumes periodic windows and can ring at the boundaries.
At 128 Hz the operation is an identity.

``normalize_signal=True`` z-scores each channel/window BEFORE resampling.
No cross-subject statistics are fitted. ``std_correction=0`` uses population
standard deviation; one uses sample standard deviation. ``std_floor`` uses
the supplied signal units. Half/bfloat16 statistics are computed in float32
then cast back. The default leaves signal normalization to the caller.
Filtering, source units and reference selection belong to the data pipeline.

``freeze_encoder=True`` freezes all encoder parameters and keeps it in
evaluation mode while the head trains. Input gradients remain available.
Otherwise existing parameter trainability flags are preserved. For example::

    backbone = STEEGFormer(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
    model = STEEGFormerClassifier(backbone, ['C3', 'C4'], chunk_size=800,
        sampling_rate=200, normalize_signal=True, freeze_encoder=True)
    logits = model(torch.randn(2, 2, 800))

Wrapper ``from_pretrained`` takes explicit ``backbone_kwargs`` and task
settings, restoring a complete encoder and initializing a fresh task head.
Native metadata/decoder handling follows the safe encoder loader above.
To restore a trained task, rebuild both configurations and strictly load its
complete state. An encoder load alone does not restore a trained task head.

Standalone workflows
---------------------

* :doc:`/auto_examples/examples_eegmat_steegformer_classifier`: background vs
  arithmetic classification with a held-out subject.
* :doc:`/auto_examples/examples_eegmat_steegformer_regression`: arithmetic-only
  prediction of published subtraction counts. At least three subjects and
  two different training targets are required. The built-in RegressorTrainer
  supports scalar targets; multiple outputs require an explicit task loss.
* :doc:`/auto_examples/examples_eegmat_steegformer_embeddings`: pooled features
  with every source metadata row in unchanged dataset order.

These use native 128 Hz, complete-recording .5--45 Hz FIR filtering,
four-second windows, source channel order/reference, per-window population
z-scoring and a frozen encoder. Epochs are fixed and subjects stay separate.
Regression records the metadata hash, fixed target scale and MAE in original
units. Every artifact includes complete tensor state, both constructor
configurations, preprocessing and all predictions/targets or feature rows.
No test target selects epochs or parameters; no benchmark result is implied.


Complete masked autoencoder
---------------------------

Construct the full Small, Base or Large model with
``STEEGFormerPretrainer.for_variant('small')``. Load every original encoder
and decoder tensor with an explicit variant::

    from torcheeg.models import STEEGFormerPretrainer

    model = STEEGFormerPretrainer.from_pretrained(
        'checkpoint-300.pth', variant='small', allow_namespace=True)
    indices = model.channel_indices(['Fp1', 'Fp2', 'Cz'])
    loss, prediction, mask = model(eeg, indices, mask_ratio=.75)
    loss.backward()

The full models have 160, 256 and 400 state tensors, respectively.
Small's decoder has width 384 and four blocks, despite the upstream factory
name saying dec256. Base and Large both use width 512 and eight decoder
blocks. All three decoders have 16 attention heads. Native MAE has no
dropout or stochastic depth; the pretrainer rejects these additions.
The fresh MAE uses the original Xavier Linear initialization and normal
CLS/mask-token initialization. Full loading rejects incomplete states; it
never drops decoder tensors. Meta construction and restricted CPU mmap
loading avoid allocating a second Large parameter set. Create optimizers
after loading; model and optimizer checkpoints remain separate concepts.

The pinned source has two different encoder temporal-position contracts.
Regular ``forward_encoder`` starts temporal patches at zero; its decoder
starts at one. This is the pretrainer default
``encoder_position_offset=0``. Original ``forward_demo`` starts encoder
patches at one: select ``encoder_position_offset=1`` explicitly for that
protocol. Inherited unmasked ``forward_all`` and ``forward_features`` keep
the existing encoder's position-one contract. An identical state dictionary
does not encode this choice; save the selected policy with experiment state.

``forward`` returns scalar masked reconstruction loss, predictions shaped
``[batch, patches * electrodes, patch_size]``, and a binary mask shaped
``[batch, patches * electrodes]``. Only mask-one patches contribute to
mean squared error. ``norm_pix_loss=True`` explicitly enables the original
unbiased per-patch target variance with epsilon 1e-6; the verified official
Small/Base training metadata uses False. ``mask_ratio`` must be in (0, 1];
one retains only CLS. The zero-mask undefined loss is rejected.
``patchify`` and ``unpatchify`` preserve temporal-major, channel-minor order.

Supply a device-compatible ``torch.Generator`` to control masking without
changing the global RNG. Save its state, full model state, optimizer state,
constructor settings, channel/preprocessing/partition settings and next batch
cursor for exact continuation. Validation must use a separate generator.

Official local checkpoint provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtain files from the author's
`release assets <https://github.com/LiuyinYang1101/STEEGFormer/releases>`_.
Verify the exact bytes before opting into native metadata. No weights ship
with TorchEEG. The fixed three-variant protocol uses Large 196, rather than
the separate expanded-channel LargeV2 artifact.

* Small: ``checkpoint-300.pth``, 393972208 bytes; SHA256
  ``98d2848abdaea7a143d399c07f67f8782e07dd66e7c309ddb058c7158a8a8b02``.
* Base: ``checkpoint-288.pth``, 1333478786 bytes; SHA256
  ``3c04469072b1e2acf9333ee48ee1002f2db42f46ffb0daab6269686ebc6fe6c1``.
* Large: ``large_weights_only_196.pth``, 1316497385 bytes; SHA256
  ``7b2ed01dec88938ea0a8aba5621eab283bb61ec12b88c6475722fffae8afda83``.

Small and Base require the scoped Namespace opt-in; Large 196 is a bare
tensor state. Restricted loading has no unsafe fallback. For an older
runtime lacking safe_globals, convert a verified full state to a tensor-only
file on a supported runtime first. Preserve every decoder key during conversion.

Real-data continuation example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`/auto_examples/examples_eegmat_steegformer_pretraining` trains the full
MAE using one-second EEGMat windows and a held-out subject. It requires
official full weights, records source hashes and protocol, saves AdamW and
masking RNG, and can resume at the next training batch. It has no random
small-model fallback. Its fixed sequential batches and small teaching run
do not reproduce the original corpus, optimizer schedule or benchmark scores.

Validation scope
~~~~~~~~~~~~~~~~

Independent float64 fixtures cover both temporal-position modes, both target
normalization policies, partial/full masking, reconstruction outputs, loss,
input gradients and every used encoder/decoder parameter gradient. The
complete official Small/Base/Large networks were also compared against the
unmodified original MAE on two real 19-channel, 128-sample EEGMat windows:
outputs, loss and input gradients matched exactly; all parameter gradients
passed rtol=1e-4/atol=1e-6 (maximum absolute difference below 2e-9).
This is CPU functional validation with full model geometry and weights,
not a large-corpus result or GPU/distributed-training acceptance.
