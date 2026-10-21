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

The classification example trains a new task head. The independent
self-supervised workflow below uses the original augmentations, span masking
and all three losses. Its teaching schedule is explicit; the original full
TUEG corpus and pretraining schedule are not reproduced.

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

Complete self-supervised training
----------------------------------

:class:`torcheeg.models.UniDMFMPretrainer` connects every original reconstruction
and contrastive head to the released training objective. Supply the complete
``UniDMFM`` adapter and an explicitly constructed native ``NTXentLoss`` module.
The library imports no author code and bundles neither source nor weights.
The standalone example obtains the loss, augmentations and span-mask sampler
from the same trusted pinned local author checkout.

.. code-block:: python

   from torcheeg.models import UniDMFMPretrainer

   # native_loss is explicitly constructed from the separately obtained source.
   pretrainer = UniDMFMPretrainer(backbone, native_loss)
   result = pretrainer(clean_eeg, first_view, second_view, mask)
   result['loss'].backward()

Clean EEG and both independently augmented views are ``[B,C,T]`` at 200 Hz.
The explicit binary ``[B,C,P]`` mask zeros only the first augmented view.
The original second augmented view supplies the other contrastive projection.
Both reconstruction targets use **clean original EEG**, even if augmentation
shifted the first view. No target is silently changed to the augmented signal.

The time loss is masked **MSE**, summed over selected samples and divided by
``mask.sum()*200 + 1e-8``. The frequency target uses periodic Hann64, hop16,
centered reflect padding, and one-sided STFT: ``log(abs(STFT)+1e-6)``, flattened
to 429 bins (33 frequencies by 13 frames). Frequency MSE is summed over selected
bins and divided by ``mask.sum()*429 + 1e-8``. It is not ``log1p`` or a raw FFT.
Only the frequency target is detached; time-target and augmented-view gradients
follow the original source. Although author variables say ``mae``, the released
criterion is squared error.

The supplied original NT-Xent normalizes the two projections, forms the full
``2B`` similarity matrix, clamps scaled similarity to ``[-50,50]``, removes its
diagonal and predicts positive pairs offset by ``B``. Its explicit temperature
is recorded with the loss coefficients. The default total is the sum of all
three losses; callers can supply nonnegative coefficients. Empty masks retain
the original zero reconstruction terms. A batch of one has zero NT-Xent signal;
use at least two examples for meaningful negatives.

All **386 native states** remain present, including 144 profiling buffers.
The physical metadata, stored float32 Hann64 window and objective fingerprint
bring the full wrapper to **389 states**. Checkpoint restoration rejects a
changed objective or temperature. A supplied fixed native window can be used
for numerical comparison and is saved exactly, avoiding reconstruction of its
values on another CPU.

Independent real-data continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`../auto_examples/examples_eegmat_unidmfm_pretraining`.
The standalone example uses all 19 EEGMat electrodes, whole-record resampling
to 200 Hz, FIR 0.1–75 Hz and 50-Hz notch, original reference, and **20-second**
windows. It converts to float32 before per-channel unbiased standardization
``(x-mean)/(std+1e-6)``, then clamps to ``[-10,10]`` as in native pretraining.
This differs from the population-standardized four-second classification probe.

The original source generates independent views through whole-window temporal
shifts, channel dropout and Gaussian noise. Its span lengths of 5–10 patches
can overlap and are clipped at the window end. ``mask_ratio=0.6`` therefore
means an approximate masking policy, not an exact 60 percent selection.
The example calls the original functions; it does not replace this sampler.

The last sorted subject is held out. Labels never enter optimization. The CPU
teaching run uses native AdamW coefficients and gradient clipping, with a fixed
1000-update cosine schedule and 100-update warmup. This schedule is independent
of ``--steps``, which is the total number of updates including resumed ones.
It is a small teaching transfer, not a TUEG/full-corpus or GPU-AMP reproduction.

.. code-block:: console

   python examples/examples_eegmat_unidmfm_pretraining.py --root-path ./eegmat \
       --backend-root ./Uni-DMFM --checkpoint ./pretrained_weights.pth \
       --steps 2 --output-path ./pretrain2.pt
   python examples/examples_eegmat_unidmfm_pretraining.py --root-path ./eegmat \
       --backend-root ./Uni-DMFM --checkpoint ./pretrained_weights.pth \
       --resume ./pretrain2.pt --steps 3 --output-path ./pretrain3.pt

The original checkpoint is required even on resume to establish the complete
profiling schema and provenance. Saved training state includes all model and
profiling tensors, window, optimizer, scheduler, Torch/Python/NumPy RNG and
next batch cursor. NumPy RNG is encoded as restricted-loadable tensors and
basic values. Validation uses a separate fixed RNG, then restores the training
streams. Input hashes, cache manifests and protocol are checked; existing output
files are rejected. Native helper namespaces are explicitly checked: use a fresh
process if a different author checkout already occupies their module names.

Normal numerical tests use clearly labeled protocol doubles and an independent
NumPy STFT reference. Optional tests use the **complete official model** and
compare all predictions, three losses, raw EEG and every parameter gradient.
They also execute the untouched original CPU training method and compare the
complete optimizer step, gradient clipping, scheduler and profiling state.
The dedicated GitHub workflow supplies the original source and checkpoint.
GPU, shared-model concurrency and distributed pretraining remain unverified.
