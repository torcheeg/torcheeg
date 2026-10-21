mdJPT with an external backend
==============================

TorchEEG provides an explicit interface to the complete original mdJPT
MLLA/CNN model. All 82 released state tensors (754,114 elements), including
the original projection layers and unused released parameters, remain
loaded. A new task head does not replace any native projection state.

Source and installation
-----------------------

See `Multi-dataset Joint Pre-training of Emotional EEG Enables Generalizable
Affective Computing <https://arxiv.org/abs/2510.22197>`_ and the
`author repository <https://github.com/ncclab-sustech/mdJPT_nips2025>`_ at
``bb84e89df5d05928ef3aa8645cbc339b59ed59de``. The pinned source has **no stated
license**. Obtain source and weights separately under appropriate terms.
TorchEEG bundles neither author implementation nor checkpoint. See
``LICENSES/MDJPT-BACKEND-NOTICE.txt`` for attribution and provenance.

The complete ``log/pretrain/ckpt/epoch=19.ckpt`` is 6,453,928 bytes with SHA256
``9a8da20e6ca4b4709251c13e16c88a155d3fa6a9b1a8411a37f019d6440e98b9``.
The original 60-by-60 ``channel_interpolate.npy`` has SHA256
``bc083c8b44d984b627d265a7ef500a9953347dfcfc7995184de0217581c6b8e2``.
Keep this physical neighbor table with its source revision.

Install ``omegaconf==2.3.0`` and the normal model/dataset dependencies,
including timm and PyTorch Lightning, in the explicit backend environment.
The example factory verifies the Git revision and unchanged tracked source,
resolves the saved original configuration, and constructs all original
neural classes. Use a fresh process if another package named ``src`` is
already loaded. Library imports never import this source or download weights.

The original constructor loads a relative interpolation-table file. The
factory redirects only that load through a module-local NumPy facade; CWD
and global NumPy are unchanged. A dependency import changes
``PYTORCH_NVML_BASED_CUDA_CHECK``; the factory restores the caller's exact
previous value. The original patch helper hardcodes a CUDA allocation.
The factory explicitly installs an independently implemented device-neutral
sliding-window helper. Original neural class bodies and all weights remain
unchanged. This is an allocation port, not an assertion that the original
unmodified helper supports CPU. The CPU oracle compares against the original
loop with only that allocation redirected; a separate actual CUDA oracle
requires real CUDA and is otherwise explicitly skipped.

Physical input and full architecture
------------------------------------

``MDJPT`` accepts finite float32 ``[batch,electrodes,625]`` EEG at **125 Hz**.
Each window is five seconds. Declare actual physical names in input order.
The native vocabulary is available as ``MDJPT.PRETRAINING_CHANNELS``:

.. code-block:: text

   FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2
   FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6
   TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 O1 OZ O2

By default all 60 physical sensors must be present. For fewer measured
sensors, explicitly set ``allow_interpolation=True``. The original helper
maps available sensors and averages up to the first three available sensors
in its fixed neighbor order for missing positions. Interpolated positions
are not newly measured electrodes. Unknown, duplicated or silently ignored
input names are rejected. Historical physical equivalents require explicit
``channel_aliases``, for example ``{'T3': 'T7', 'T4': 'T8'}``.
The EEGMat example's T3/T4/T5/T6 correspondences follow
`standardized 10-10/10-20 nomenclature <https://www.acns.org/UserFiles/file/EEGGuideline3Montage.pdf>`_.
Both actual names and declared equivalences are saved with the configuration.

The full neural architecture includes two width-128, eight-head MLLA layers,
32-sample windows with stride six, and 32-wide outputs at 99 temporal
positions for every native sensor. The native channel projection maps 60 to
64 positions. The complete multiscale spatial/temporal CNN has 512 feature
channels. Its final projector returns normalized 1024-wide features.
Both original attention paths, positional convolutions, feed-forward layers
and all released states are kept, including source parameters unused by a
particular feature view. A parameter's presence does not imply that every
objective uses it.

Ordinary independent inference
------------------------------

The source normalizes across windows within each half-batch in both
training and evaluation. Consequently an ordinary batch with four or more
rows can change one window's output when another same-group window changes.
A one-row native call fails. This grouping must not be hidden in a standard
sample classifier.

``model(eeg)`` explicitly calls the complete source in two-row chunks. Each
native half-group contains one actual window. An odd final singleton is
duplicated and its second output discarded. The unchanged released neural
configuration has zero dropout. This gives per-window statistics, sample
isolation and output invariance to batch partitions. It is an explicit
transfer protocol, distinct from original multiwindow paired normalization.
It does not create additional training examples or positive pairs.

Choose a fixed ``feature_mode`` when constructing the adapter:

* ``'mean'`` returns the mean original pre-projector map ``[B,512]``.
* ``'de'`` returns ``0.5*log(2*pi*e*population_variance+1)`` over that map's
  temporal positions, ``[B,512]``, matching the native posthoc DE formula.
* ``'projector'`` returns the original normalized 1024-wide projector output.

For mean/DE modes, ``return_tokens=True`` returns the native feature map
``[B,512,99]``. These 512 channels are learned features, not electrode names.
The native feature-map/projector branch is fixed once at construction;
forward calls do not mutate shared source flags. Construct separate complete
backends for different fixed modes. All 82 original states remain in each.
The full paper's running normalization, LDS smoothing, ICA and downstream
emotion task protocol are not inferred by these feature views.

``MDJPTClassifier`` adds a new linear task head. With
``freeze_backbone=True`` every original parameter remains frozen in evaluation
mode while the new head trains. Otherwise gradients follow the chosen full
feature path, with originally frozen and unused states retained as released.

Explicit native paired training
-------------------------------

``forward_native`` takes ``[2,windows,electrodes,625]``. Each first-dimension
entry identifies a distinct explicit normalization group. Windows within a
group share native statistics. It returns summarized ``features``, complete
``covariance_features`` and ``channel_tokens``, plus ``feature_map`` or
``projector`` according to the fixed mode. Pair/group identities are caller
responsibilities and are never inferred from class labels.

In projector mode, ``forward_pretraining(domain_batches)`` calls both original
training loss objects. Supply a nonempty list of explicit paired domain
tensors. Each has shape ``[2,pairs,C,625]``; positive pairs occupy corresponding
ranks in the two groups. The original contrastive top-five report requires
at least three pairs. For different physical montages, a domain can be a
dictionary with ``eeg``, ``channel_names`` and optional ``channel_aliases``.
Interpolation intent remains explicit for every domain.

The output contains summed ``contrastive_loss``, unweighted
``alignment_loss`` and aggregate ``loss``. The aggregate applies the saved
native CDA factor once; backpropagate ``loss`` once. Native CDA retains the
author helper's center/transpose/reshape convention. It is not silently
replaced by a different channel covariance estimator. Grouping can use other
unlabeled windows, so this explicit protocol differs from sample-independent
holdout inference. Do not mix held-out labels into pair construction.

Safe loading and full restoration
---------------------------------

Construct the explicit trusted backend, then call
``MDJPT.from_pretrained(path, backend, channel_names, ...)``. The official
Lightning checkpoint uses the default ``state_key='state_dict'`` and empty
prefix. Complete flat exports use ``state_key=None``; common prefixes are
explicit. Safetensors exports also use a flat tensor dictionary.
Restricted CPU ``weights_only=True`` loading validates all keys, shapes,
dtypes and finite values before any tensor load. There is no unsafe Lightning
pickle fallback, partial projection selection or silent architecture change.

Physical names/order/aliases, interpolation table, fixed feature mode,
source grouping/device policy and loss/dropout/normalization configuration
contribute to the saved fingerprint. Strict restoration checks it even after
``to_empty``. Preserve original source/config/table/checkpoint hashes,
preprocessing, group identities and optimizer/RNG/data position with a
training export.

Runnable EEGMat workflow
------------------------

:doc:`../auto_examples/examples_eegmat_mdjpt` uses all 19 measured EEGMat
sensors and declares T3/T4/T5/T6 equivalences explicitly. It opts in to native
interpolation for 41 missing native positions. Whole recordings are
resampled from 500 to 125 Hz, filtered at 1--47 Hz and re-referenced to the common average
over the 19 measured sensors. Five-second windows use channelwise population
z-scores with a 1e-6 microvolt standard-deviation floor. Background is 0;
mental arithmetic is 1. No ICA or fitted target-population normalization is
applied. These are explicit teaching transfer choices.

The last complete subject is held out. Only a new head over independent
mean features trains. All native weights, physical/protocol configuration,
source hashes, labels and predictions are exported and strictly restored.
A real three-subject run used 96 training windows, 48 held-out windows and 12 head
updates. Its 75% accuracy equals the 75% majority baseline: it is a functional
teaching run, not a published emotion-recognition benchmark reproduction.
Gallery rendering does not execute this training workflow.

Independent functional tests require no author code. The complete native
CPU allocation-port oracle requires ``TORCHEEG_MDJPT_BACKEND_ROOT`` and
``TORCHEEG_MDJPT_CHECKPOINT``. It compares mean, DE and projector outputs,
independent/native grouping, both native objectives and input/every used
parameter gradient in evaluation and training. A separate CUDA oracle has
an explicit hardware skip. Local documentation success does not establish
a Read the Docs cloud build or unmodified GPU validation.

Actual three-window paired noisy-view training was also checked: all 74 used
native parameters received finite gradients and updated. Complete 83-state
model/native Adam/augmentation RNG restoration reproduced the next objective
and every next state exactly. These positive pairs are two noisy views of
the same real window, not the paper's original cross-subject stimulus pairs.
The complete original pretraining dataset and schedule remain separate.

Modern compilation and numerical limits
---------------------------------------

On the verified PyTorch 2.6 Mac CPU runtime, default Inductor with
``fullgraph=False`` preserves ordinary mean features and input/every used
parameter gradient within 1e-5 relative/1e-6 absolute tolerance. This executes
the complete neural model; no reduced network or error-suppression fallback
is substituted. A working C++ toolchain is required. Original scalar accuracy
reporting and input validation can cause graph breaks.

The full native two-domain pretraining objective is more sensitive to FP32
rounding. In the tested case, compiled versus ordinary execution had about
0.092% relative L2 input-gradient difference and 0.106% parameter-gradient
difference; componentwise 1e-5/1e-6 agreement failed. A separate double-precision
mathematical diagnostic measured parameter-gradient relative L2 errors of
0.047% for compiled FP32 and0.094% for ordinary FP32. This diagnoses finite
precision; it does not establish universal error bounds or change the
released model's precision. The eager original-source oracle still agrees
exactly in outputs and used parameter gradients on the same CPU runtime.

PyTorch 2.6 ``aot_eager`` backward fails on a native attention view/stride
operation. Static input-contiguity and identity-gradient experiments did not
resolve it and are not part of the public model. Use ordinary execution for
the verified native objective/continuation protocol. Bitwise compiled
pretraining equivalence, unmodified GPU execution, fullgraph compilation and
GPU performance are not established by the CPU checks.
