Pretrained YASA sleep staging
=============================

:class:`torcheeg.models.YASASleepStaging` provides the complete optional YASA
traditional sleep-staging baseline. It runs the original contextual feature
extractor and pretrained LightGBM classifier, rather than substituting a neural
network or training a new classifier from a few epochs. See the
`YASA documentation <https://yasa-sleep.org/generated/yasa.SleepStaging.html>`_
and the `method publication <https://doi.org/10.7554/eLife.70092>`_.

Install and version scope
-------------------------

Use an isolated Python 3.11+ environment::

   pip install "torcheeg[yasa]"

For this feature branch, install ``-e ".[yasa]"`` from the checkout.
The adapter verifies YASA **0.8.1** and the exact eight official classifier
files distributed with that PyPI version. Other YASA versions have different
interfaces and are rejected explicitly. YASA is loaded only during prediction;
ordinary TorchEEG imports and installations do not acquire a TensorFlow/YASA
runtime dependency or a higher Python floor.

YASA 0.8.1 is BSD-3-Clause. No YASA implementation or classifier is copied into
the TorchEEG distribution. The optional dependency supplies its official
models. Each file's pinned SHA256 is checked before YASA loads its joblib.
There is no arbitrary classifier path or automatic model download argument.
Retain the YASA publication citation when using its predictions.

Continuous waveform input
-------------------------

Supply one continuous MNE Raw recording in volts. Use a central EEG derivation
such as C4-M1 where available; EOG and chin EMG are optional. Selected channels
must exist, be distinct and have MNE voltage channel types. The original
0.8.1 implementation requires a sampling rate strictly greater than 80 Hz.
It internally downsamples to 100 Hz and converts volts to microvolts.

Do not standardize, filter or manually convert the input to microvolts first.
Do not call the model independently on each 30-second window: its complete
feature context uses neighboring epochs, including a centered 7.5-minute
feature window and a past two-minute window. This is offline, future-context
inference. At least one complete epoch is required; the upstream recommendation
is at least five minutes. The adapter preserves upstream short-record messages.

.. code-block:: python

   import mne
   from torcheeg.models import YASASleepStaging

   raw = mne.io.read_raw_edf('recording.edf', preload=True)
   model = YASASleepStaging(
       eeg_name='C4-M1', eog_name='EOG', emg_name='EMG',
       metadata={'age': 30, 'male': True},
   )
   result = model.predict(raw)

Demographic metadata is optional. Supply both ``age`` and ``male`` or neither,
to match the full demographic classifier feature set. Use only appropriate
available metadata; it is copied and never inferred from labels. Calling the
same adapter again extracts fresh features from the current waveform, without
caching by object identity. The original Raw data, channels and sampling rate
are not changed.

Stable class order and physical times
-------------------------------------

YASA 0.8.1 returns a Hypnogram object. The adapter maps its ``WAKE/N1/N2/N3/REM``
values to TorchEEG's explicit ``W/N1/N2/N3/REM`` order, categorical IDs zero
through four. Both labels and probability columns use that order. The
LightGBM classifier's alphabetical internal class order is not assumed.

``predict`` returns a dictionary:

* ``labels``: int64 array ``[epochs]``.
* ``probabilities``: float64 array ``[epochs, 5]``.
* ``onsets``: seconds from the first supplied sample, spaced by 30 seconds.
* ``recording_onsets``: onsets plus ``raw.first_time``, retaining a cropped
  recording's physical origin.
* ``stage_names`` and ``metadata``: explicit class order, channels, rates,
  units, classifier filename/hash, backend version and protocol provenance.

Incomplete trailing epochs are discarded by the original extractor. This
traditional estimator is not an ``nn.Module``; it has no gradients, local
supervised ``fit`` or optimizer state. YASA's internal ``fit`` extracts
features and does not train the pretrained classifier.

Verification and known scope
----------------------------

Tests execute all eight original pretrained combinations of EEG, optional
EOG/EMG and demographic features. Every complete backend's probabilities and
labels are compared directly with the unmodified YASA API. Other cases cover
raw preservation, cropped origins, resampling/tails, fresh extraction,
metadata, channel/rate/units/finite boundaries and classifier integrity.
No feature extractor or classifier is stubbed in numerical verification.

The official pretrained artifacts include a LabelEncoder serialized with
scikit-learn 0.24.2. The tested current 1.9.1 runtime emits an upstream
serialization-version warning. That warning remains visible; direct complete
classifier parity and real-record execution are checked on the recorded
runtime. This is not a guarantee for arbitrary future dependency combinations.

Real Sleep-EDF example
----------------------

See :doc:`../auto_examples/examples_sleepedf_yasa`. The example reads the full
Fpz-Cz EDF in MNE volts and predicts before reading any Hypnogram annotations.
It adds no waveform preprocessing and fits no local statistics. Only afterward
does evaluation align complete physical 30-second annotation intervals,
merge original stages 3/4 as N3 and retain movement/unscored gaps as ``-1``.

It saves safe numeric NPZ arrays and JSON provenance atomically without
replacing an existing result. Files load with ``allow_pickle=False``.
Single-record forehead-channel transfer is a functional teaching case, not
YASA's preferred central montage, a clinical validation or a paper/Arena
leaderboard result. Keep original source files and model provenance with any
reported result.
