ADFTD resting-state EEG
=======================

Source and task
----------------

:class:`torcheeg.datasets.ADFTDRSDataset` reads the eyes-closed routine EEG
release `OpenNeuro ds004504 v1.0.8
<https://openneuro.org/datasets/ds004504/versions/1.0.8>`_.
The metadata snapshot is Git commit
``73b1b04c0a8e45e96c6f0682cc077a7919480fc4`` in
`OpenNeuroDatasets/ds004504
<https://github.com/OpenNeuroDatasets/ds004504/tree/73b1b04c0a8e45e96c6f0682cc077a7919480fc4>`_.
Cite Miltiadous et al., *A Dataset of Scalp EEG Recordings of Alzheimer's
Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG*,
Data 8(6), 95 (2023), https://doi.org/10.3390/data8060095.

The release has 88 participants: 36 AD, 23 FTD and 29 healthy controls.
The diagnosis mapping is ``C -> HC (0)``, ``A -> AD (1)``, ``F -> FTD (2)``.
The participants table supplies these labels; no label is inferred from
age, MMSE or signal values. The photic-stimulation dataset ds006036 is a
separate task. Its provider states that participant numbers match this
cohort, so do not treat equal IDs across the two tasks as independent people.

Point ``root_path`` at the full BIDS root. The default reads each matching
``sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set``. Matching session/run files
are retained as separate records. A known participant missing the requested
recording or matching channel/JSON sidecars raises an error. Invalid or
duplicate participant IDs and unknown diagnosis codes also raise errors.
The parser accepts the release's text BOM and surrounding header whitespace.

Channels and physical units
----------------------------

The output order matches ``ADFTD_RS_CHANNEL_LIST``:
``FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ``.
The SET labels determine the input order and are checked/reordered into this
canonical uppercase order before preprocessing. Require all 19 recorded EEG
channels even when selecting fewer output channels. The source rate is
500 Hz and each SET is a continuous, single-trial array. Embedded MATLAB
arrays and matching external float32 FDT data are supported. Malformed
headers, nonfinite data and inconsistent time/duration metadata fail.

The source channel table labels values ``microV``. EEGLAB stores microvolt
numbers; this reader converts them to **volts** and sets ``signal_unit='V'``.
Keep this unit convention distinct from a separate hardware calibration.
Source channel-unit metadata are retained as JSON. The SET reference field
is ``A1 A2``; the README also describes a Cz acquisition montage. The reader
does not resolve that historical wording by silently applying a new reference.

Preprocessing and source selection
-----------------------------------

The default settings retain the original addition's requested pipeline:
50 Hz notch, 0.5--45 Hz FIR, output rate 200 Hz and 400-sample windows
(two seconds), with no additional reference. These are explicit extra
processing steps, not a claim to reproduce the provider's ASR/ICA pipeline.
The raw SET JSON describes software filtering, so ``raw`` here means the
release's original branch, not an assertion of an untouched amplifier signal.

Each contiguous segment is processed with all 19 channels: notch, FIR,
optional average reference, then resampling. Only then are the first
``num_channel`` output channels selected. To preserve native released
samples, set ``sfreq=500``, ``notch_freqs=None``, ``l_freq=None`` and
``h_freq=None``. Warnings from short-segment filtering are not suppressed.

Set ``preprocessed=True`` to select the provider's ``derivatives/sub-XXX``
SET files. Participant labels and original channel/JSON sidecars still come
from the BIDS root. The flag changes source selection only: configured extra
filtering and resampling still apply. Use the native settings above to keep
the cleaned released numbers apart from conversion to volts. The provider
reports Butterworth filtering, re-referencing, ASR and ICA/ICLabel cleaning;
this reader loads that result and does not rerun or claim to reproduce it.

ASR can remove time intervals. A SET ``boundary`` event with one-based
latency ``n + 0.5`` marks a cut after sample ``n``. Filter and resample each
segment separately. Windows never cross a cut. A nonpositive ``chunk_size``
returns each complete contiguous segment, rather than joining segments over
removed data. Incomplete last windows of each segment are omitted. The
reader preserves the stored, compressed signal clock; it does not reconstruct
the original clock from ASR removal durations.

Load native windows
--------------------

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import ADFTDRSDataset

    dataset = ADFTDRSDataset(
        root_path='./ADFTD-RS',
        sfreq=500,
        notch_freqs=None,
        l_freq=None,
        h_freq=None,
        chunk_size=500,
        overlap=0,
        preprocessed=False,
        io_path='./cache/adftd-rs-original-native-1s',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'),
    )
    eeg, label = dataset[0]  # [19, 500], in volts

Metadata retain integer ``subject_id`` and full BIDS ``participant_id``,
source level, task, session, diagnosis and source/segment-qualified
``record_id``/``trial_id``/``clip_id``. Original and cleaned records retain the
same subject ID but distinct record IDs. Keep all such records/windows from
a participant together in evaluation partitions.

``start_at`` and ``end_at`` are half-open sample bounds in the concatenated
output representation. ``source_start_seconds`` and ``source_end_seconds``
refer to the corresponding interval within the **stored** source SET,
including when the source is the provider's compressed derivative.
``segment_index`` identifies the contiguous interval; ``source_clock`` makes
the stored-clock convention explicit. Channel names and source channel-unit
metadata are JSON fields.

Hooks and overlapping windows receive independent arrays. A ``before_trial``
hook must preserve finite segment geometry. Overlap must be nonnegative and
smaller than each window; invalid settings fail before creating a cache.
Use a new cache path when changing source level or preprocessing; see
:doc:`cache`. Grid, adjacency and region constants follow the same canonical
order and are templates, not measured patient connectivity.

Signal-loading checks do not establish classification performance. Keep test
labels out of preprocessing/model/epoch selection. A standalone training
workflow and any verified performance report require separate execution.


Regional electrode groups
--------------------------

The general, frontal and hemisphere region lists contain indices into the
unchanged 19-channel source order. For matching the shared regional templates,
``T3/T4/T5/T6`` use the equivalent modern names ``T7/T8/P7/P8``. Each family
includes all 19 source columns once. Signal labels, grid locations and both
adjacency matrices retain their original values. These regional templates
are anatomical priors, not measured patient connectivity.
