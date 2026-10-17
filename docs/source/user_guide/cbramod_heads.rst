CBraMod downstream tasks
========================

Use :class:`torcheeg.models.CBraModClassifier` for class logits,
:class:`torcheeg.models.CBraModRegression` for continuous predictions and
:class:`torcheeg.models.CBraModEmbedding` for features. Each takes an explicit
:class:`torcheeg.models.CBraMod` backbone and fixed input geometry. No data or
weights are downloaded. A random backbone is not a pretrained representation.

Input geometry and physical time
--------------------------------

These wrappers accept ``[batch, num_electrodes, chunk_size]``. The final axis
is divided into ``num_patches`` equal, nonoverlapping segments. Each segment
is independently resampled with :class:`torcheeg.models.FourierResample` to
``backbone.patch_size`` samples. The chunk size must be divisible by the patch
count. This retains the original workspace's per-segment input adaptation
while correcting its Nyquist-bin interpolation.

For example, four seconds at 256 Hz means 1024 raw samples. With four patches
and the canonical 200-sample backbone, each one-second segment becomes 200
samples. The output still represents four seconds at 200 Hz. Resampling an
entire four-second window as one period generally gives different boundary
values. Fourier interpolation assumes periodic segments; endpoint jumps can
cause ringing. See :doc:`fourier_resampling`.

Configure channel order, reference, sampling rate, physical duration,
filtering and signal units in the data pipeline. The native author backbone
was trained on 200-sample, one-second patches in units of 100 microvolts.
Resampling alone does not recover missing electrodes or establish equivalent
pretraining preprocessing. There is no learned channel mixing in these wrappers.
The examples state their explicit preprocessing differences from pretraining.

Task heads and features
-----------------------

Let ``H = backbone.hid_channels``. The default classifier/regressor flattens
all electrode/patch features and applies ``4H -> H -> outputs``, with ELU and
configured dropout after each hidden layer. This follows the author's
``all_patch_reps`` head. Other native heads are explicit configurations:

.. list-table:: Head configurations
   :header-rows: 1

   * - Author head
     - pooling
     - head_channels
   * - avgpooling_patch_reps
     - mean
     - ()
   * - all_patch_reps_onelayer
     - flatten
     - ()
   * - all_patch_reps_twolayer
     - flatten
     - (H,)
   * - all_patch_reps
     - flatten
     - None, equivalent to (4H, H)

Other positive hidden-width tuples are explicit custom heads. To reproduce
the archived workspace's wider hidden layer for more than four patches, set
``head_channels=(max(4, num_patches) * H, H)``. The old unused ``proj_dim`` and
``seq_len`` arguments are not accepted. Dropout now uses its actual argument;
the archived implementation hardcoded 0.1. Constructor names follow the rest
of TorchEEG rather than adding aliases for the unreleased workspace interface.

Classification returns unnormalized ``[batch, num_classes]`` logits.
Regression returns ``[batch, num_outputs]`` without target scaling or output
activation. Set units explicitly and fit population target statistics on
training data only. The current ``RegressorTrainer`` supports a scalar
sample target and uses output column zero; use another training loop for
multiple simultaneous targets.

``feature(x)`` returns the four-dimensional backbone representation before
pooling/head dropout; ``feature(x, flatten=True)`` flattens it. Embedding
forward flattens by default and accepts ``flatten=False`` at construction.
The native reconstruction projection is retained in the complete state but
is not used to extract task features. Optional masks use the backbone's native
``[batch, electrodes, patches]`` mask convention.

Freezing and checkpoints
------------------------

With ``freeze_encoder=True``, backbone parameters are frozen and its dropout
remains in evaluation mode during head training. Input gradients still pass
through the backbone. With freezing disabled, existing parameter flags are
preserved; the wrapper does not silently unfreeze supplied components.

``from_pretrained(path, backbone_kwargs, ...)`` safely loads a complete LOCAL
backbone tensor state, requiring the reconstruction projection as well. The
head starts from random weights. Supply an explicit prefix for a nested
``state_dict`` checkpoint. This does not load a trained task model. To restore
one, save both constructor configurations, rebuild the wrapper, then use
strict ``load_state_dict`` with a tensor state loaded using ``weights_only=True``.
Record preprocessing and channel order separately; parameter shapes cannot
prove them. Source licenses and weight provenance are described in
:doc:`cbramod`.

Standalone examples
-------------------

* :doc:`/auto_examples/examples_eegmat_cbramod_classifier`: fixed-epoch,
  subject-held-out classification with the native multi-layer head.
* :doc:`/auto_examples/examples_eegmat_cbramod_regression`: published continuous
  subtraction-count targets, arithmetic recordings only, subject isolation
  and a fixed target scale.
* :doc:`/auto_examples/examples_eegmat_cbramod_embeddings`: ordered frozen
  feature extraction with source row metadata and complete restoration settings.

Each accepts optional complete local author weights and otherwise uses a
small random backbone. These are teaching/functional protocols, not benchmark
scores or validated clinical predictors. No score is added to the leaderboard.
