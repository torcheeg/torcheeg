NeuroGPT with an external backend
=================================

TorchEEG provides an explicit interface for complete separately supplied
NeuroGPT weights. The native Conformer, embedder, causal GPT and unembedder
remain present, including all 193 released state tensors (79,528,857 elements).
No encoder-only checkpoint or replacement GPT is used.

Source and installation
------------------------

See the `paper <https://arxiv.org/abs/2311.03764>`_,
`author repository <https://github.com/wenhui0206/NeuroGPT>`_ at
``230571d45ca4369b82f33ab42fc00863c2f95598``, and the
`official weight release <https://huggingface.co/wenhuic/Neuro-GPT>`_ at
``2e0b940580db95e6237227c27c13b2319909d0fc``.
Author source and weights specify **GPL-3.0**. Obtain them separately under
their own terms. TorchEEG bundles neither implementation code nor weights.
See ``LICENSES/NEUROGPT-BACKEND-NOTICE.txt`` for attribution and provenance.

The file ``pretrained_model/pytorch_model.bin`` is 318,189,905 bytes with SHA256
``2ea7d931a812baf107cdfb41bf0f7b5568043ae2758e30b4b1b0e9fae3ff3f1a``.
This value also matches the provider's LFS object hash.

The explicit example factory constructs all original author classes from
a trusted pinned local checkout. It verifies the Git revision and tracked
source state, and refuses to replace unrelated loaded packages named
``encoder``, ``embedder`` or ``decoder``. Use a fresh process if these names
already belong to other code. Source module registration lets PyTorch locate
classes during compilation. Library imports do not import this source.

Install ``torchinfo==1.8.0`` and ``transformers==4.57.6`` in the isolated
backend environment alongside the normal model/dataset dependencies.
The original source specifies Transformers 4.38.1; that historical dependency
runtime is not established by the current validation. The current factory
explicitly selects the original eager GPT2 attention protocol in both the
configuration and the model's cached implementation setting, retaining all
weights and original model/helper bodies.

Physical input and complete architecture
-----------------------------------------

``NeuroGPT`` requires floating inputs ``[batch,segments,22,500]`` at **250 Hz**.
Each segment is a complete **two-second EEG window**. Supply one to 512
segments for feature extraction and at least two for causal reconstruction.
Segment boundaries, overlap, reference, filtering, units and scaling remain
explicit caller responsibilities. There is no automatic resampling, temporal
repetition, channel interpolation or padding.

Declare exactly 22 distinct physical channel names in their actual input
order. The primary pretraining batcher's explicit order is available as
``NeuroGPT.PRETRAINING_CHANNELS``:

.. code-block:: text

   FP1 FP2 F7 F3 FZ F4 F8 T1 T3 C3 CZ C4 T4 T2 T5 P3 PZ P4 T6 O1 OZ O2

By default that exact physical order is required (case-insensitive).
For a different montage or order, explicitly set
``allow_montage_transfer=True``. This acknowledges a different transfer
protocol; it does not interpolate or relabel sensors. The BCI-IV 2a example
sets this flag because its physical montage differs. The declaration is
saved with physical metadata and checked on restoration.

The spatial convolution operates on all 22 ordered ranks; weight
shapes do not establish a global physical montage. Preserve the montage
required by the chosen data protocol. Do not silently rename sensors or
supply 19 EEG channels as a 22-channel input.

The complete released architecture is:

* A six-layer, width-40, ten-head Conformer, producing 27 by 40 values per
  segment, including its original convolution and batch-normalization state.
* The original 1080-to-1024 embedder and native masking/CLS state.
* A six-layer, width-1024, 16-head GPT with 512 positions, inner dimension
  4096 and ``gelu_new`` activation, including the original pooler states.
* The complete 1024-to-1080 reconstruction unembedder.

The native pretraining path does not use the released CLS, vocabulary and
pooler parameters. They remain loaded and restored; their presence alone
does not establish that they are trained classification features.

Feature extraction and task training
-------------------------------------

``forward_encoder`` returns native Conformer features ``[B,S,27,40]``.
``forward(return_tokens=True)`` passes those features through the complete
original embedder and unmasked causal GPT, returning ``[B,S,1024]``.
The default output is the final actual segment feature ``[B,1024]``.
This explicit feature view adds no new CLS or pooler activation and does not
invoke pretraining prefix masking.

``NeuroGPTClassifier`` adds a new linear task head. With
``freeze_backbone=True`` every native parameter and buffer remains frozen
in evaluation mode while that new head trains. Otherwise gradients update
the complete feature backbone. The author's downstream decoding setup adds
its own CLS/pooler/classification path; this new task head is a different
explicit transfer protocol.

Causal attention applies within the GPT. In evaluation mode, future raw
segments and other examples cannot affect past valid features. In training
mode, the original Conformer's batch normalization collects statistics over
encoded segments and examples. Thus the entire raw-EEG training pipeline is
not a strictly online causal system. Padding is unsupported; fabricated
segments would also affect that normalization.

Native causal reconstruction
-----------------------------

``forward_reconstruction`` calls the complete original ``compute_loss``
with actual segments and native valid attention. It constructs every next-
segment prefix and returns ``prediction``, ``reconstruction_loss`` and ``loss``.
For ``B`` examples with ``S`` segments, predictions have shape
``[B*(S-1),S,1080]``. Targets are detached native Conformer features, not raw
EEG values. Prefix rows follow the original example/prefix ordering.
Backpropagate ``loss`` once: ``reconstruction_loss`` names the same component,
not an additional term. Prefix expansion and attention can require substantial
memory for long sequences; the position limit is not a memory guarantee.

For an explicitly constructed and fully loaded model:

.. code-block:: python

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   model.train()
   optimizer.zero_grad()
   result = model.forward_reconstruction(eeg_segments)
   result['loss'].backward()
   optimizer.step()

Save optimizer state, normalization buffers, RNG state and data position when
resuming stochastic pretraining. This interface does not establish a
reproduction of the entire original TUH pretraining schedule.

Safe complete loading and restoration
--------------------------------------

Construct the trusted local backend, then call
``NeuroGPT.from_pretrained(path, backend, channel_names)``. The release is a
flat state dictionary, so the default ``state_key=None`` is correct. Complete
nested exports can specify ``state_key`` and an explicit common ``prefix``.
Safetensors exports are also supported. All keys, shapes, dtypes and finite
values must match before any load. There is no unsafe pickle fallback,
partial matching, head removal or positional-table resizing. Torch files
use restricted ``weights_only=True`` CPU loading with memory mapping.

Physical names and order, sample rate, segment size, GPT configuration,
dropout settings and causal-prefix protocol contribute to a saved fingerprint.
Strict restoration checks it even after ``to_empty``. Keep source revision,
checkpoint hashes, preprocessing and segment-onset rules alongside states.
Do not switch the native model to encoder-only or newly initialized decoding
mode while using this interface.

Runnable BCI-IV 2a workflow
---------------------------

:doc:`../auto_examples/examples_bciciv2a_neurogpt` uses actual 22-channel
BCI-IV 2a/BNCI 001-2014 MATLAB recordings at 250 Hz and their original left
mastoid reference. It excludes EOG and marked artifact trials, filters whole
EEG runs at 0.5--40 Hz and extracts the four-second cue interval (trial onset
plus two seconds). See :doc:`bciciv2a` for correct one-based MATLAB conversion.

Each trial/channel is standardized by its population mean and standard
deviation, with a 1e-6 microvolt floor. Two nonoverlapping 500-sample segments
form each example. The last complete subject is held out; only a new
four-class head trains. Labels map original 1--4 to 0--3 (left hand, right
hand, feet, tongue). Complete 196-state model/configuration/provenance exports
are strictly restored and held-out predictions checked.

This is a short-context teaching transfer example. Original pretraining uses
32 segments with 50-sample overlap, whole-record channel normalization
before segment selection, and a different downstream decoding head. A real run with 543 training and 270 test trials
was independently verified from raw MAT data and direct native inference.
The frozen probe's 24.44% accuracy is below the 25.56% majority baseline;
it is not a published benchmark reproduction or a competitive result.
Gallery rendering does not execute the training workflow.

Independent functional tests need no author source. A separate native oracle
requires ``TORCHEEG_NEUROGPT_BACKEND_ROOT`` and
``TORCHEEG_NEUROGPT_CHECKPOINT``. It compares complete native encoder/GPT
features, reconstruction outputs/losses, input/every parameter gradient and
running statistics in evaluation and stochastic training modes. Without
these explicitly supplied paths, only that native oracle is skipped.
Local documentation success does not establish a Read the Docs cloud build.

PyTorch 2.6 CPU ``aot_eager`` compilation with ``fullgraph=False`` preserves
feature/reconstruction outputs, input/every parameter gradient and running
statistics in training mode. Native prefix expansion and validation require
graph breaks; fullgraph compilation and GPU performance are not established.
