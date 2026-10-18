LaBraM downstream tasks
=======================

:class:`torcheeg.models.LaBraMClassifier`,
:class:`torcheeg.models.LaBraMRegression` and
:class:`torcheeg.models.LaBraMEmbedding` wrap the complete
:class:`torcheeg.models.LaBraMEncoder` described in :doc:`labram_encoder`.
The existing LaBraM classification API remains available separately.

Input and physical duration
----------------------------

Supply real floating ``[batch, electrodes, chunk_size]`` windows, explicit
source ``sampling_rate`` and ordered ``channel_names``. The wrapper computes
an exact native 200 Hz sample count from physical duration, then requires
complete backbone patches. Partial samples, unknown channel identities and
inconsistent lengths are errors. The backbone's explicit temporal-table
interpolation option applies when a window exceeds its learned table.

``resample_mode='window'`` Fourier-resamples the complete window as a single
period. ``resample_mode='patch'`` independently resamples equal source
segments to native patches, preserving the archived workspace operation.
For a canonical encoder, a four-second 128 Hz window has 512 source samples
and becomes four 200-sample patches. Independent patch mode requires that
the source divides exactly into equal segments. It can differ numerically
from whole-window resampling at boundaries. Both methods assume periodic
signals and can ring at discontinuities. At native 200 Hz no interpolation
is needed.

Filtering, reference choice, input units and amplitude normalization remain
in the data pipeline. Native fine-tuning divides microvolts by 100. The
wrappers do not z-score signals or fit population statistics.

Token features and task heads
------------------------------

``feature(eeg)`` returns the channel-major grid
``[batch, electrodes, patches, hid_channels]``, excluding CLS.
``feature(eeg, flatten=True)`` flattens it from axis one.
Default ``normalize_features=True`` applies the complete native learned
final norm separately to every token. False exposes raw final-block tokens.

Per-token norm before pooling differs from the encoder's learned norm
AFTER raw-token mean pooling. It also differs from the author's separately
initialized fine-tuning fc_norm. The wrappers explicitly reuse the complete
encoder norm and do not silently rename or omit its pretrained weights.
The default reproduces the archived all-token/flatten topology with this
complete norm restoration, not a claim of reproducing a published task head.

Classification produces unnormalized ``[batch, num_classes]`` logits.
Regression retains ``[batch, num_outputs]`` even for scalar targets and
applies no output activation. Set target units/scaling explicitly. The
built-in RegressorTrainer supports scalar targets; multiple output dimensions
require an explicit task loss.

Default ``pooling='flatten'`` and ``head_channels=None`` select::

    electrodes * patches * H -> max(4, patches) * H -> H -> outputs

Each hidden linear layer has ELU and configurable dropout. An explicit
head_channels tuple chooses other hidden widths; an empty tuple chooses a
linear output without hidden dropout. ``pooling='mean'`` averages the selected
token features over channels and time before that head. It does not change
the selected normalization order.

Embedding extraction has no learned task head. Default ``flatten=True``
returns ``[batch, electrodes * patches * H]``; False retains the feature grid.
Random encoder state is not a pretrained representation.

Freeze and restoration
----------------------

``freeze_encoder=True`` freezes all backbone parameters and keeps it in
evaluation mode while a task head trains. Input gradients remain available.
Otherwise existing parameter trainability flags are preserved. For example::

    backbone = LaBraMEncoder()
    classifier = LaBraMClassifier(backbone, ['Fp1', 'C3', 'C4'], chunk_size=512,
        sampling_rate=128, num_classes=2, resample_mode='window', freeze_encoder=True)

For fixed EEG window geometry with Fourier operations, compile with
``torch.compile(classifier, fullgraph=True, dynamic=False)``. FFT kernels in
the tested runtimes require concrete dimensions. This preserves full-graph
training and does not imply symbolic FFT support.

Wrapper ``from_pretrained`` requires explicit ``backbone_kwargs`` and all
sampling/task settings. Native student prefix, recognized pretraining-head
removal and scoped metadata opt-in follow the encoder's safe loader. This
restores a complete encoder and initializes a fresh task head. To restore
a trained task, reconstruct both configurations and strictly load its whole
tensor state. No remote download or unsafe pickle fallback occurs.

Standalone workflows
---------------------

* :doc:`/auto_examples/examples_eegmat_labram_classifier`: background versus
  arithmetic classification, holding out a whole subject.
* :doc:`/auto_examples/examples_eegmat_labram_regression`: published subtraction
  counts from arithmetic windows. At least three subjects and two different
  training targets are required; divide by fixed factor 10 and report MAE in
  original metadata units.
* :doc:`/auto_examples/examples_eegmat_labram_embeddings`: channel-major
  flattened features and every source metadata row in unchanged dataset order.

These use complete-recording 200 Hz resampling, .1--75 Hz bandpass and 50 Hz
notch MNE FIR, four-second windows, source channel order/reference and
microvolts divided by 100. They select per-token native norm, whole-window
resampling (an identity at that rate) and a frozen encoder. Defaults use a
small random encoder; the verified local author base checkpoint is optional.
No data or weights are downloaded.

Classification/regression fix epochs and keep held-out labels out of model
selection. Regression stores its metadata hash and fixed target scale.
Artifacts save complete encoder/head tensor state, both constructor settings,
preprocessing/channel/partition details and all outputs/targets or ordered
features/metadata. These are teaching runs, not benchmark reproductions or
leaderboard claims. Choose new caches when inputs or offline filters change.
