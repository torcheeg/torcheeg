LUNA pretrained models
======================

:class:`torcheeg.models.LUNA` implements the complete pretraining network from
`LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis
<https://github.com/pulp-bio/BioFoundation>`_ (NeurIPS 2025). It retains the
reconstruction decoder and fixed channel embeddings, loads every official
weight, and exposes temporal features for downstream tasks.
:class:`torcheeg.models.LUNAClassifier` adds the native learned aggregation
query, attention and MLP classification head. Its head is newly initialized.

All three variants
------------------

.. list-table::
   :header-rows: 1

   * - Variant
     - Query width
     - Queries
     - Temporal depth
     - Feature width
     - Checkpoint tensors
   * - base
     - 64
     - 4
     - 8
     - 256
     - 224
   * - large
     - 96
     - 6
     - 10
     - 576
     - 254
   * - huge
     - 128
     - 8
     - 24
     - 1024
     - 464

All variants use patch size 40 and two heads per query feature width; temporal
attention has ``2*num_queries`` heads. These dimensions match the actual
released reconstruction states. Parameter counts for those complete states
are 6,672,417 / 40,660,337 / 303,769,569; networks built with a
classification head have different parameter counts.

Download the corresponding safetensors file explicitly from
`author revision 999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f
<https://huggingface.co/PulpBio/LUNA/tree/999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f>`_
and verify SHA256:

* ``LUNA_base.safetensors``:
  ``482839ad9152b6948ff166d2a1638637c54b4d91f1cefadbce6312831ba9b11d``.
* ``LUNA_large.safetensors``:
  ``03e0321a1e1a30dc074f7bfe512a901138200494f7694f31ab633e42b105c0e4``.
* ``LUNA_huge.safetensors``:
  ``7522e85c6403675bff3c1b3856e5e93b724ac0da4d7d0e7e11bfba9a57067346``.

Install ``safetensors`` or ``pip install 'torcheeg[foundation]'`` for the safe
local loader. It never downloads or executes remote code and never falls
back to pickle. Key sets, shapes, dtypes and finite values must all match;
missing decoder or channel-embedding weights are errors. An optional explicit
prefix selects a complete state within a larger safetensors dictionary.
``drop_path`` can configure temporal stochastic depth without changing tensor
shapes; the default is zero, and the source fine-tuning YAML uses 0.1.

Signals, coordinates and outputs
--------------------------------

The canonical source protocol uses 256 Hz, five-second segments, 0.1--75 Hz
bandpass filtering, a selected 50/60 Hz notch, and per-channel/window z-score
normalization. Declare the source reference, physical units and exact order
of preprocessing operations. The model performs none of those signal steps.

Supply floating EEG ``[batch, electrodes, samples]``, with a nonempty sample
axis divisible by ``patch_size``. Supply coordinates in one shared coordinate
frame as ``[electrodes, 3]`` or ``[batch, electrodes, 3]``. The model normalizes
coordinates per example and axis before sinusoidal position embedding. It
does not infer electrode identities or repair a mismatched coordinate order.

Temporal convolution and rFFT magnitude/phase form patch features. Learned
queries cross-attend to the channel axis, and rotary transformer blocks mix
the resulting temporal sequence. The feature path supports channels outside
the fixed reconstruction vocabulary because it only needs their coordinates.

.. code-block:: python

    from torcheeg.models import LUNA, LUNAClassifier

    backbone = LUNA.from_pretrained('./LUNA_base.safetensors', variant='base')
    backbone.eval()
    features = backbone.forward_features(eeg, positions)  # [B, patches, 256]
    ids = LUNA.get_channel_ids(['FP1', 'C3', 'PZ'])
    reconstructed = backbone(eeg, positions, ids)  # [B, 3, samples]
    model = LUNAClassifier(backbone, num_classes=2, freeze_backbone=True)
    logits = model(eeg, positions)  # [B, 2], use cross entropy directly

``get_channel_ids`` maps case-insensitive names to the fixed, ordered 90-row
pretraining vocabulary. It does not build bipolar derivations or estimate
coordinates. IDs can be shared ``[electrodes]`` or per-example
``[batch, electrodes]`` integer tensors. Unknown reconstruction names are
rejected rather than mapped to an arbitrary trained row.

``return_attention=True`` returns a pair: features or reconstruction, and
channel-unification attention ``[batch, patches, queries, electrodes]``.
For one fixed dataset layout, bind coordinates with
``LUNAClassifier(backbone, positions=positions)``. They are copied into a
model buffer, move with ``to()`` and are included in ``state_dict``. The model
then accepts ordinary ``model(eeg)`` calls through ``ClassifierTrainer``.
Per-call coordinates override that buffer for heterogeneous batches.

Classification uses a learned pooling query and the native dropout-0.15 head.
Setting ``freeze_backbone=False`` enables full backbone fine-tuning; the
retained reconstruction components are unused in classification forward.

Masking and separate examples
-----------------------------

Boolean ``mask`` accepts patch shape ``[batch, electrodes, patches]`` or sample
shape matching the EEG. Any selected sample masks its entire patch embedding.
This preserves native masking after convolutional GroupNorm: raw masked
samples can still influence visible-token normalization statistics. It is
feature masking, not a promise of raw-sample isolation or a complete
self-supervised training objective. Callers that require input corruption
must define that protocol explicitly.

Position jitter is explicit: ``position_noise=0.02`` requests the source's
Gaussian augmentation after coordinate normalization. The default zero is
deterministic in evaluation, including masked evaluation. This differs from
the author's unconditional noise whenever a mask argument is supplied.

The adapter also fixes the native per-example repeat order. A heterogeneous
batch uses each example's own coordinates and decoder IDs for every patch;
its results agree with running those examples individually. Both position
embeddings and reconstruction channel embeddings use this corrected order.
Native parity checks use single-example or homogeneous metadata, while
heterogeneous behavior has a separate functional and gradient test.

There are three independent workflows, with fixed variants and no model sweep:

* ``examples/examples_eegmat_luna_base.py``.
* ``examples/examples_eegmat_luna_large.py``.
* ``examples/examples_eegmat_luna_huge.py``.

Each requires the complete matching local checkpoint, retains all 19 EEGMat
scalp channels, computes explicit MNE standard1005 head-frame coordinates,
and trains a frozen-backbone classifier. The reader resamples complete
recordings to 256 Hz, then MNE FIR bandpass/notch filtering runs on complete
recordings, followed by five-second windowing and per-channel z-score. This
explicit after-resampling teaching protocol does not reproduce source
pipelines that filter first. The last sorted complete subject is held out;
all other subjects train for a fixed epoch count, without test-based selection.

The saved artifact contains every model tensor, constructor/preprocessing
JSON, coordinate tensor, source/checkpoint hashes, split IDs, targets and
predictions. Each script safely loads and strictly restores the full model,
then requires exact held-out predictions before reporting success. Short runs
verify functionality, not the paper's benchmark performance.

Sources, licenses and verification
----------------------------------

Source implementation: BioFoundation revision
``d4484feccf62eeec666380cb55b5d5b1d517cd71``, Apache-2.0. MLP operations follow
timm 1.0.10 and rotary operations follow rotary-embedding-torch 0.8.9; notices
and licenses are retained in ``LICENSES/LUNA.txt``. Runtime model computation
needs neither timm nor rotary-embedding-torch.

Official weights separately use CC-BY-ND-4.0, as stated in the
`pinned author model card
<https://huggingface.co/PulpBio/LUNA/blob/999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f/README.md>`_.
That card describes internal fine-tuning and restrictions on sharing modified
weights. Follow the original terms for any local exports. No author weights
or adapted validation checkpoints are bundled in TorchEEG or its examples.

Independent fixtures compare full reconstructions, attention and every
EEG/coordinate/parameter gradient against the unchanged author implementation.
Portable training references control scalar dropout, disable attention-
probability dropout, and suppress coordinate jitter; real stochastic behavior
is tested separately. Full official states of all three canonical variants
were also compared against the author model on PyTorch 2.6 CPU; every measured
output and gradient maximum error was zero. These checks, actual EEG teaching
runs, package tests, local documentation, GitHub CI and Read the Docs cloud
builds are recorded independently.
