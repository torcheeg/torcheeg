AD-Auditory BIDS data
=====================

``ADAuditoryDataset`` reads the released 40 Hz auditory entrainment EEG in
OpenNeuro ds005048 v1.0.0. See the
`dataset <https://openneuro.org/datasets/ds005048/versions/1.0.0>`_ and
`source paper <https://doi.org/10.1038/s41598-024-63727-z>`_.
Install ``torcheeg[dataset]`` for the HDF5 header dependency.
This reader targets the released continuous MATLAB v7.3 SET/FDT format.
It is not a general EEGLAB reader for embedded data or epoched SET files.

Source and labels
-----------------

The release has 35 participants. ``participants.tsv`` supplies diagnosis groups;
this adapter maps Normal to HC (0), Mild AD and Moderate AD together to AD (1),
and MCI to MCI (2). The two unknown ``'-'`` rows are excluded. The labeled
cohort therefore has 10 HC, 17 AD and 6 MCI participants. These are diagnosis
targets; Stimulus and Rest describe recording conditions, not diagnosis labels.

Each selected participant needs its matching SET, FDT and events TSV files
under ``sub-*/eeg/`` or ``sub-*/ses-*/eeg/``. Multiple matching task recordings
are all retained. Pairing follows the complete filename, including run/session
entities. Missing known-participant recordings, duplicate participant IDs,
unknown group spellings and mismatched metadata raise errors.

The SET header supplies the sampling rate, channel labels, point count and
external FDT filename. The reader verifies the continuous 19-channel, 250 Hz
layout and exact FDT byte count, and restores the canonical channel order
from header labels. It does not infer geometry from a binary file's length.
Signals remain in microvolts following the EEGLAB external-data convention;
the released channels TSV lists units as ``n/a``, so this is a format convention,
not an independently calibrated ADC-unit claim. Optional MNE processing receives
volts internally and returns microvolts.

The source README describes provider preprocessing: high-pass and line-noise
filtering, bad-channel handling/interpolation, average reference, ASR and ICA
cleaning. Default TorchEEG reading adds no filter, reference or resampling.
Optional processing is additional processing of an already cleaned release;
it does not reproduce the paper's full cleaning or analysis pipeline.

Events and windows
------------------

The event table gives onset and duration in seconds. Stimulus intervals last
40 seconds; Rest intervals last 20 seconds, except the final 5-second Rest.
Recordings contain six or ten stimulus intervals, with actual lengths of
350 or 590 seconds. The first five seconds lie outside these annotations.
The matching one-based ``sample`` column is checked against onset seconds;
it is not used as a zero-based Python index.

By default, ``trial_types=('Stimulus', 'Rest')`` selects annotated intervals.
Choose ``['Stimulus']`` or ``['Rest']`` to restrict conditions. Only explicit
``trial_types=None`` uses the whole recording, including unannotated time;
that mode can read a recording without an events table. Missing, empty,
overlapping, nonfinite or out-of-range event tables otherwise raise errors.
They do not trigger whole-record fallback or silent interval clipping.

``chunk_size`` and ``overlap`` count output samples. Nonpositive chunk sizes
use each complete selected interval. Positive chunks discard only their final
incomplete remainder, not the rest of the recording. ``sfreq`` is an integer
output rate. Event seconds are rounded onto that actual output sample grid.
Notch/FIR, average reference and resampling operate on each complete recording
before interval extraction; average reference uses all 19 released EEG channels
before the requested output subset is selected.

``before_trial`` receives an independent copy of a selected interval and must
preserve finite real-valued geometry. Each overlapping window is copied before
its offline transform. Source arrays and neighboring windows remain independent.

Usage and evaluation
--------------------

.. code-block:: python

   from torcheeg import transforms
   from torcheeg.datasets import ADAuditoryDataset

   dataset = ADAuditoryDataset(
       root_path='./AD-Auditory',
       trial_types=['Stimulus'],
       chunk_size=250,
       io_path='./cache/ad-auditory-stimulus-native',
       online_transform=transforms.Compose([
           transforms.To2d(), transforms.ToTensor()]),
       label_transform=transforms.Select('label'))

Each item returns a signal and its diagnosis label, with shape ``[1, 19, 250]``
in this example. For additional 128 Hz processing, explicitly pass ``sfreq=128``
and choose an output chunk such as ``chunk_size=128`` for one second.
Select a fresh cache path when changing sources, conditions or preprocessing.

Metadata includes participant, complete recording ID, optional BIDS session ID
(``n/a`` when absent), a recording-qualified original event ID, diagnosis/group,
condition/value, output sample positions, actual output-grid times, original
event seconds, sample rate, channel names and unit convention. Restricting to
Rest retains the original event indices. Different recordings and participants
do not share trial or clip IDs.

Keep every recording, condition and window from a person in the same partition
for unseen-person evaluation. Split by ``subject_id`` before model selection.
Use validation participants for settings and hold out test participants for the
final evaluation. Random sample splits can mix repeated windows from the same
participant. This reader establishes data access and label mapping; it does not
establish a clinical prediction result or a published benchmark score.

Layout constants
----------------

``AD_AUDITORY_CHANNEL_LIST`` and ``AD_AUDITORY_CHANNEL_LOCATION_DICT`` retain the
19 mixed-case EEG names and grid order from the original implementation.
``AD_AUDITORY_ADJACENCY_LIST`` and ``AD_AUDITORY_ADJACENCY_MATRIX`` provide its
hand-defined electrode graph. These constants describe a model layout;
they are not subject-measured functional connectivity.


Standalone EEGNet training
--------------------------

The gallery provides :doc:`../auto_examples/examples_ad_auditory_eegnet`.
Run it on the downloaded, provider-cleaned release:

.. code-block:: console

   python examples/examples_ad_auditory_eegnet.py --root-path ./AD-Auditory --epochs 1 --batch-size 64 --output-path ./ad-auditory-eegnet.pt

The script uses all 19 EEG channels, native 250 Hz and nonoverlapping
one-second windows from both annotated conditions. It applies per-window,
per-channel normalization, with no additional filter, reference or resampling
and no statistics fitted across people. A constant channel becomes finite
zeros. The diagnosis label is already 0/1/2; condition values are not targets.

For each diagnosis class, hold out the last sorted participant ID and train
on the others. Every record, condition, event and window from that participant
stays in the held-out partition. The script requires all three classes and
at least two participants per class. In the full 33-person labeled release,
this gives 30 training people and three test people. The JSON records actual
IDs, source labels, records, represented event counts and complete-window
counts by condition. Participant selection uses labels for stratification
and does not use model predictions.

Training uses a fixed epoch count with no test-based model or epoch selection.
Reported accuracy/F1 and cross-entropy pool the test windows. This is one fixed
illustrative split; it provides neither participant-voted metrics nor full
cross-validation. Retain that scope when reporting the output. A clinical or
published benchmark result needs its own complete protocol and validation.

The export includes all nine EEGNet constructor settings, all 20 state tensors,
and preprocessing/partition JSON strings. It freezes constrained convolution
weights and checks exactly equal inference after strict in-memory and safe
``weights_only=True`` disk restoration. Existing model and sidecar files are
preserved. Use a fresh cache after changing input data or offline processing.
The gallery renders the script without executing its training run.
