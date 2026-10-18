SleepDG sequence architecture
==============================

:class:`torcheeg.models.SleepDG` provides the native sequence encoder,
projected features, reconstruction decoder and per-epoch classifier from
`Wang et al., AAAI 2024 <https://arxiv.org/abs/2401.05363>`_. It follows the
MIT-licensed `author implementation <https://github.com/wjq-learning/SleepDG>`_
at revision 01d209d4fda874933a277a88bccdca6e3835e884. It retains native
parameter and BatchNorm-buffer names without adding a model dependency.
No trained checkpoint is downloaded or bundled.

Input and sequence boundaries
------------------------------

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
-------------------

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
-----------------------------

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
-------------------------------

The model exposes the tensors required by the complete SleepDG objective:
per-epoch cross-entropy, signal reconstruction MSE, epoch feature-distribution
alignment and sequence-relation alignment. These losses are external to the
architecture. Classification plus reconstruction alone is a supervised
component check; it does not establish cross-dataset domain generalization.
For those two components::

    from torch.nn import functional as F

    logits, reconstruction, features = model.forward_all(eeg)
    classification = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    reconstruction_loss = F.mse_loss(reconstruction, eeg)

Flatten epoch logits and targets consistently. The ordinary ClassifierTrainer
expects its standard flat-sample contract and does not infer this sequence
axis or multi-domain objective. Keep source-domain membership and target
holdout isolation explicit in a complete training workflow.

Save the full constructor configuration and complete state, including native
encoder/decoder buffers and the trained classifier. Rebuild the same geometry
before strict loading. Encoder-only or classifier-only tensors do not restore
a trained full model. A random model is not a pretrained sleep representation,
and synthetic correctness checks are not sleep-staging performance evidence.
