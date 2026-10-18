SleepDG sequence architecture
=============================

:class:`torcheeg.models.SleepDG` provides the native sequence encoder,
projected features, reconstruction decoder and per-epoch classifier from
`Wang et al., AAAI 2024 <https://arxiv.org/abs/2401.05363>`_. It follows the
MIT-licensed `author implementation <https://github.com/wjq-learning/SleepDG>`_
at revision 01d209d4fda874933a277a88bccdca6e3835e884. It retains native
parameter and BatchNorm-buffer names without adding a model dependency.
No trained checkpoint is downloaded or bundled.

Input and sequence boundaries
-----------------------------

Supply real floating [batch, sequence_length, num_electrodes, 3000] epochs.
Native settings use 20 consecutive 30-second epochs sampled at 100 Hz,
with one EEG and one EOG channel in an explicit order. The Transformer
operates across these epochs. It does not treat convolutional positions
within one epoch as the sleep-stage sequence.

Do not cross a recording, subject, dataset partition or an annotation gap
when forming a sequence. Each epoch needs its own target. Preserve native
sampling, EEG/EOG order, reference and signal units in preprocessing. The
model performs no filtering, resampling or normalization of raw signals.
The native decoder reconstructs 3000 samples; arbitrary chunk lengths are
rejected rather than cropped or padded into a different reconstruction.

The author's Sleep-EDF preparation selects EEG Fpz-Cz and EOG horizontal,
filters at .3--35 Hz, merges sleep stages 3/4 into N3, and maps W/N1/N2/N3/REM
to 0/1/2/3/4. It standardizes channels over each retained recording, including
unlabeled target-recording statistics. This differs from fitting normalization
on training subjects alone. Record the chosen protocol explicitly and do
not claim the author's domain-generalization scores for a different split
or preprocessing scheme.

Native computation
------------------

Four shared Conv1d/BatchNorm/GELU/max-pooling stages encode each epoch,
using widths 64, 128, 256 and hid_channels. Adaptive average pooling removes
the remaining within-epoch time axis. A learned-position sequence Transformer
uses one block, eight heads and width/MLP512 by default. The fc_mu projection
produces one feature per epoch. It is deterministic, with no variational
sampling or KL term in the author's active model.

The classifier produces five unnormalized logits per epoch. The decoder
starts each projected feature at length1 and uses six ConvTranspose1d
layers to reconstruct its two 3000-sample signals. Five decoder stages
include BatchNorm/GELU/dropout. The default dropout is .1; a separate
attention probability dropout may be configured.

sequence_length, num_electrodes and hid_channels can be configured, with
corresponding state shapes. Native author state requires sequence20,
channels2, width512, depth1, heads8 and MLP512. Smaller configurations are
extensions for explicit experiments and tests, not pretrained variants.
Single-batch/single-epoch/single-hidden-width configurations retain every
axis. Construction performs no dummy forward or BatchNorm update.

Inference and reconstruction
----------------------------

For ordinary inference::

    from torcheeg.models import SleepDG
    import torch

    model = SleepDG()
    model.eval()
    with torch.no_grad():
        logits = model(eeg)                    # [B, 20, 5]
        features = model.feature(eeg)          # [B, 20, 512]

forward follows the author's inference path and skips the decoder.
feature returns the native projected fc_mu features. feature_dim() returns
the per-epoch hidden width. To request the complete author model path::

    logits, reconstruction, features = model.forward_all(eeg)
    # [B, 20, 5], [B, 20, 2, 3000], [B, 20, 512]

The decoder updates BatchNorm buffers and uses training dropout only when
that path is requested in training mode. Standard train/eval calls control
all registered modules. Copying, device/dtype moves and strict tensor-state
restoration include the decoder and all BatchNorm buffers.

Training components and scope
-----------------------------

The model exposes the tensors required by the complete SleepDG objective:
per-epoch cross-entropy, signal reconstruction MSE, epoch feature-distribution
alignment and sequence-relation alignment. These losses are external to the
architecture. Classification plus reconstruction alone is a supervised
component check; it does not establish cross-dataset domain generalization.
Use the complete loss with explicit source membership::

    from torch.nn import functional as F
    from torcheeg.trainers import SleepDGAlignmentLoss

    # domains: [B], integer source-dataset codes in [0, num_domains).
    alignment = SleepDGAlignmentLoss(num_domains=4)
    logits, reconstruction, features = model.forward_all(eeg)
    classification = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    reconstruction_loss = F.mse_loss(reconstruction, eeg)
    loss = classification + reconstruction_loss + alignment(features, domains)
    loss.backward()

Flatten epoch logits and targets consistently. The ordinary ClassifierTrainer
expects its standard flat-sample contract and does not infer this sequence
axis or multi-domain objective. SleepDGTrainer implements the training phases and default schedule described
below. Data preparation and a standalone multi-dataset example remain separate
work. Keep target features and labels
out of the source-alignment call. In the published protocol, each domain is
a different dataset; treating subjects from one dataset as domains is a
different experiment.

Alignment scales and edge cases
-------------------------------

The loss follows the fixed author implementation, including its reduction
scales: for every unordered pair of present source domains, add the mean
squared difference between epoch means and the mean squared difference
between unbiased epoch covariances. Sum those epoch terms over domain pairs.
For sequence alignment, center each epoch across its hidden features and
compute its Pearson relation with every other epoch. Average relation matrices
within each source domain, then average squared differences over domain pairs
and sum over sequence-matrix elements. The pair average applies only to this
sequence term. It must not be applied to the epoch terms.

Available source codes are ``0`` through ``num_domains - 1``. Configure the
number of sources explicitly; absent codes in a batch are allowed and excluded
from the pair count. At least two sources must be present for a nonzero
alignment signal. A single source returns a differentiable zero. A batch with
only one sequence takes this zero path directly, keeping its input gradient
zero and its domain-code lookup checked without a covariance backward graph. Integer
lookups reject negative and oversized domain codes in eager and tested CPU
compiled/exported paths. The module owns no learned parameters or persistent
statistics. Save ``num_domains`` and ``eps`` with the training configuration.

A source with one epoch has zero covariance in this implementation. Constant
epoch vectors have zero correlations, using a lower bound ``eps`` (default
``1e-12``) on the product of centered feature norms. These finite conventions
extend the author code, which has undefined singleton/zero-variance cases.
Ordinary nondegenerate inputs retain the author loss and input gradients.
FP16/BF16 features use FP32 statistics outside autocast; FP64 inputs retain
FP64. This does not normalize the input EEG or make every half-precision loss
scale safe.

Training with SleepDGTrainer
----------------------------

:class:`torcheeg.trainers.SleepDGTrainer` trains with per-epoch cross-entropy,
reconstruction MSE and multi-level alignment. Source training batches contain
``(eeg, labels, domains)``. Validation and test batches contain ``(eeg, labels)``
or the same three-tuple; evaluation ignores domain codes, calls encoder-only
classification, and does not compute reconstruction or alignment.

Before creating loaders, assemble each sequence from consecutive 30-second
epochs in one recording. Check true start/end times and record identifiers;
never bridge movement/unscored gaps or join different nights. Keep all nights
of a held-out person out of training when that protocol requires person
isolation. Map source-dataset names to integer codes from zero through
``num_domains - 1``. Target features must not enter the source training loader,
and target labels must not select checkpoints or hyperparameters.

Use a source validation split for checkpoint selection::

    from pytorch_lightning.callbacks import ModelCheckpoint
    from torcheeg.models import SleepDG
    from torcheeg.trainers import SleepDGTrainer

    model = SleepDG()
    trainer = SleepDGTrainer(model, num_classes=5, num_domains=4,
                            metrics=['accuracy', 'f1score'])
    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max',
                                 save_top_k=1)
    trainer.fit(source_train_loader, source_val_loader, max_epochs=50,
                callbacks=[checkpoint])
    results = trainer.test(target_test_loader,
                           ckpt_path=checkpoint.best_model_path)

This code assumes correctly prepared loaders; it does not download or silently
prepare a dataset. A standalone multi-dataset data workflow remains separate
work. Supply the actual selected checkpoint path: each ``test`` call constructs
a new Lightning evaluator, which cannot infer a previous callback's ``'best'``
checkpoint. A test call without ``ckpt_path`` evaluates the current model.

The trainer's default Adam learning rate is ``0.001``, weight decay ``0.0001``,
and epsilon ``1e-8``. Set weight decay to ``lr / 10`` explicitly if changing the
learning rate while following the author's rule. Label smoothing is ``0.1``
for cross-entropy in every labeled phase. Reconstruction and combined alignment
weights are both ``1.0``. The fixed author code uses the same unit coefficients;
the paper also describes separately weighted terms. This trainer exposes
``weight_reconstruction`` and a combined ``weight_alignment`` coefficient.

The fit wrapper defaults to 50 epochs and gradient-norm clipping at ``1.0``.
Override Lightning options explicitly for another protocol. Cosine annealing
steps per successful optimizer update, with a horizon estimated from Lightning's optimizer
step count. This accounts for accumulation and configured training limits.
Set ``scheduler_steps`` to a planned positive horizon when splitting a run or
using a loader whose length is unavailable. Set ``lr_scheduler=False`` to use
constant-rate Adam. If AMP skips an update after nonfinite gradients, the
learning rate also stays unchanged; Lightning's global step can still count
that attempt. Thus the scheduler need not reach its horizon if updates were
skipped. These settings do not reproduce the paper's data sampling
or establish published accuracy by themselves.

Loss and classification statistics weight batches by the number of labeled
epochs, so a smaller last batch cannot receive a full batch's weight. The
training loss is the weighted average of per-step composite losses, not a
single global covariance loss over an entire epoch. Validation/test loss uses
classification only. Configured multiclass metrics flatten all sequence epochs
consistently. FP16/BF16 CE, MSE and alignment statistics use FP32; FP64 loss
inputs preserve their precision. Follow the actual tested precision and loss
scales when changing optimizer epsilon or native model dtype.

The training, source-checkpoint selection and full optimizer/scheduler resume
paths are also tested with PyTorch 1.13.1 and Lightning 1.9.5 on Python 3.10.
The optimization hooks forward the installed Lightning version's arguments,
including the extra optimizer index in Lightning 1.x. CPU float16 GradScaler
and compilation tests require newer PyTorch APIs and are skipped in that old
environment. This check does not establish support for every older dependency
combination; use precision and device options supported by your installation.

Training calls ``model.forward_all``; compile that callable when compiling the
complete architecture. The validation/prediction path uses ``model.forward``.
For compiled FP64 CPU execution, select the public math SDPA backend with
``torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`` around forward/backward.
PyTorch 2.6 CPU flash SDPA can otherwise fail with a Double/Float backward dtype
mismatch. This is a compiled math backend, not an eager fallback. FP32 keeps
normal backend selection.

For a strict CPU eager/compiled gradient comparison, use matching reference
convolution kernels with ``torch.backends.mkldnn.flags(enabled=False)`` around
both executions. The default optimized oneDNN path remains available and is
also tested for complete finite training updates. In the tested small-model
FP32 compound objective, the default server CPU kernels had a relative gradient
difference of about ``5.8e-4`` despite closely matching losses; matching reference
kernels reduced this to about ``6.9e-6``. These results do not establish bitwise
parity for every optimized kernel. The checks retain full-graph Inductor
compilation, raw per-parameter gradient checks, FP64 reference comparisons and
the actual norm-clipped FP32 gradients; no eager fallback or relaxed parity
threshold is used. Backend contexts restore the caller's settings on exit.

``verbose=False`` follows the base trainer's convention and suppresses metric
logging, while still updating statistics. Keep logging enabled when monitoring
``val_accuracy`` in a checkpoint callback. The model, optimizer and scheduler
can resume from a trusted Lightning checkpoint with
``trainer.fit(..., ckpt_path=path)``. Rebuild the matching model geometry and
use the same planned schedule. Checkpoint state restoration does not guarantee
exact mid-epoch sample continuation: use end-of-epoch checkpoints or a resumable
data loader to avoid replaying samples after an interruption.

Save the full constructor configuration and complete state, including native
encoder/decoder buffers and the trained classifier. Rebuild the same geometry
before strict loading. Encoder-only or classifier-only tensors do not restore
a trained full model. A random model is not a pretrained sleep representation,
and synthetic correctness checks are not sleep-staging performance evidence.
