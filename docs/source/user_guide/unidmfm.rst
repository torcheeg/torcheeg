Uni-DMFM with an external backend
=================================

TorchEEG provides an explicit interface for the complete separately supplied
Uni-DMFM base model. It retains time reconstruction, frequency reconstruction
and contrastive projection heads. Author implementation code and pretrained
weights are not included in the package or downloaded during imports.

Source and complete checkpoint
-------------------------------

See the `author repository <https://github.com/ichenws/Uni-DMFM>`_ at
``522f9fc09b6d5f3d1051e6941d1bff548d8f10de`` and the
`paper <https://doi.org/10.1016/j.patcog.2026.114221>`_. The author README
links a separate
`checkpoint <https://drive.google.com/file/d/1Dlk6X3YYmJfOjNKu7xaehzLFAIgnATIM/view>`_.
The pinned source and linked weight release have **no stated license**.
Obtain and use those materials separately under their applicable terms.
See ``LICENSES/UNIDMFM-BACKEND-NOTICE.txt`` for attribution and provenance.

The official ``pretrained_weights.pth`` is 353,402,773 bytes with SHA256
``92fc5f59cd3ffc1c25898f1a573db8d8194935a163bbcb9c68d45f86ba81d5fe``.
It has 242 neural state tensors plus 144 scalar, double-precision
``total_ops``/``total_params`` profiling buffers: 386 tensors and 29,421,093
elements in total. The author's fine-tuning script removes profiling keys.
TorchEEG's explicit example factory instead verifies the full original file,
registers only those nonnegative scalar counters at existing original modules,
and retains all 386 tensors. It changes no neural computation.

Install ``timm`` separately in the isolated author-backend environment.
The example factory verifies the pinned Git revision and tracked source
state, imports the original ``models/model.py`` explicitly and registers that
module so PyTorch can locate it during compilation. The library interface
accepts an already constructed trusted ``nn.Module``; it imports no author
code itself. Backend construction and restricted tensor loading are separate
actions.

Physical inputs and native architecture
---------------------------------------

``UniDMFM`` accepts floating ``[batch,electrodes,time]`` signals at **200 Hz**.
Signals must contain one to 100 complete **200-sample, one-second patches**.
Declare between one and 256 distinct physical channel names in input order.
Native positional embeddings represent **ordered ranks**, not physical
coordinates or a global electrode vocabulary. The caller must preserve and
record electrode order, reference, filtering, amplitude units and scaling.
No implicit resampling, coordinate inference or channel reordering occurs.

The complete native architecture uses three convolution scales (15, 49 and
101 samples), adaptive pooling, width 384 and twelve factored-attention
layers with twelve heads and groups of four adjacent channel ranks.
Group normalization mixes the native sensor/time layout. Padding can change
valid outputs and is **not supported**. Do not insert fake sensors or padded
time patches and interpret them as missing observations. Different real
window lengths are valid within the documented range and can change context.

``forward(return_tokens=True)`` returns native contextual token grids
``[batch,electrodes,patches,384]``. The default is their explicit mean
``[batch,384]`` for transfer learning. The author's downstream classifier
instead flattens all tokens into a fixed-window constrained linear layer.
Use token output to implement that different protocol. Mean reduction is
not described as a reproduction of the published classifier.

All three original pretraining heads
-------------------------------------

``forward_pretraining`` calls the original complete ``mode='full'`` path and
returns:

* ``time_prediction``: ``[batch,electrodes,patches,200]``.
* ``frequency_prediction``: ``[batch,electrodes,patches,429]``.
* ``contrastive_projection``: ``[batch,256]``.

``forward_contrastive`` calls the native contrastive-only path. Both
reconstruction heads and the projection remain in the state and receive
ordinary gradients when their outputs contribute to a loss. Supply explicit
augmented or masked inputs, reference targets and the appropriate losses.
This interface does not silently select augmentations or a pretraining
schedule. Native ``pretrain.py`` uses **MSELoss** for both reconstructions,
despite naming its variables ``loss_mae``. Its frequency target is detached
log STFT magnitude with FFT 64, hop 16 and a Hann window, giving 33 by 13
values per one-second patch. Its contrastive criterion normalizes projections
and uses paired-view cross entropy with a specified temperature.

The current standalone example trains a new task head. A complete independent
real-data reproduction of the original pretraining schedule, span masking
and augmentations remains separate work; output availability alone does not
establish that reproduction.

Safe full loading and metadata
-------------------------------

Construct the explicit matching author backend first, then call
``UniDMFM.from_pretrained(path, backend, channel_names)``. The default
``state_key='model'`` matches the release. Flat tensor exports, including
safetensors, require ``state_key=None``. A supplied common ``prefix`` must
match every tensor. All keys, shapes, dtypes and finite values are checked
before any state is loaded. There is no unsafe pickle fallback, partial
load, head removal or silent profiling-buffer discard. CPU loading uses
``weights_only=True`` and memory mapping for Torch files.

A saved fingerprint records ordered physical names, sample rate, patch size,
STFT settings, per-layer drop-path rates and the complete state schema.
Strict restoration rejects mismatches, including after ``to_empty``.
Record preprocessing and provenance alongside states. A state file does not
encode a complete evaluation protocol.

``UniDMFMClassifier`` is a new linear head over mean native tokens.
``freeze_backbone=True`` keeps the entire native model in evaluation mode
and freezes all native parameters; only the new head trains. Otherwise the
complete native backbone can fine tune. This head differs from the author's
fixed-window flattened-token constrained classifier.

Runnable EEGMat workflow
------------------------

:doc:`../auto_examples/examples_eegmat_unidmfm` explicitly maps all 19 physical
EEGMat channels, retains their original reference and holds out a complete
subject. It resamples full records from 500 to 200 Hz, then applies 0.1--75 Hz
FIR and 50 Hz notch filtering. Four-second windows use per-channel population
z-scores with a 1e-6 microvolt standard-deviation floor. It trains a new
mean-token head, exports all 389 model states with metadata, and strictly
restores held-out predictions.

This teaching transfer protocol differs from the author's filter-before-
segmentation, first/last 60-second selection and flattened-token task head.
It does not claim published benchmark performance. Gallery rendering does
not execute training. Run the standalone script on actual data separately.

Independent interface and workflow tests use small protocol fixtures.
The optional complete native oracle requires
``TORCHEEG_UNIDMFM_BACKEND_ROOT`` and ``TORCHEEG_UNIDMFM_CHECKPOINT``. It
compares native token/mean features and all three outputs, raw-input gradients
and every parameter gradient in evaluation and stochastic training modes.
Without the separately obtained source and weights, only that oracle skips.
Local documentation success does not establish a Read the Docs cloud build.

The real EEGMat export was independently verified from raw EDFs against
direct original-model calls. Full native tensors and EDF hashes remained
unchanged during the frozen probe. PyTorch 2.6 CPU ``aot_eager`` compilation
with ``fullgraph=False`` preserves feature and three-head outputs and all
input/parameter gradients. These checks establish neither fullgraph/GPU
performance nor the published pretraining and downstream benchmark scores.
