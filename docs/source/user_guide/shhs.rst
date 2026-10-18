SHHS recordings, stages and visits
=================================

:class:`torcheeg.datasets.SHHSDataset` reads the Sleep Heart Health Study
EDF recordings and NSRR XML annotations. Obtain data access from
`NSRR <https://sleepdata.org/datasets/shhs>`_ before using the reader.
Source data remain in their original directories. The class does not
request access, download restricted data or read the different Profusion
XML format. The `provider's polysomnography guide
<https://sleepdata.org/datasets/shhs/pages/05-polysomnography-introduction.md>`_
describes both annotation formats.

File pairs and participant identity
----------------------------------

Set ``root_path`` to the SHHS directory or its ``polysomnography`` subdirectory::

    SHHS/polysomnography/
        edfs/shhs1/shhs1-200001.edf
        annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml
        edfs/shhs2/shhs2-200001.edf
        annotations-events-nsrr/shhs2/shhs2-200001-nsrr.xml

Use ``visits=[1]``, ``[2]`` or ``[1, 2]``. An absent visit is allowed when it
has neither signals nor annotations. Selected visits must have complete
pairs. Missing partners and malformed record identifiers raise an error;
files are never paired by directory position.

``subject_id`` is the six-digit participant identifier. It is the same in
both visits. ``session_id`` is the visit number. ``record_id`` and ``trial_id``
include the visit prefix; ``clip_id`` also includes the epoch index. Split
by participant to keep repeated visits in the same evaluation partition.
A split by record alone does not provide participant isolation.

Signals and preprocessing
-------------------------

The default channels are ``['EEG', 'EOG(L)']`` in that order. Names must
match the actual EDF labels. The second EEG channel can be called
``EEG(sec)``, ``EEG2``, ``EEG 2``, ``EEG(SEC)`` or ``EEG sec`` in individual
records. Choose its recorded name explicitly when using that channel;
missing names raise an error. The reader does not invent a reference or
replace a missing channel with zeros.

MNE reads only the selected channels, aligns different native rates to their
highest rate, applies the optional FIR filter, then resamples to ``sfreq``.
EEG/EOG voltage values are in volts, rather than the microvolt-valued output
of :class:`torcheeg.datasets.P2018Dataset`. Channel order is preserved.
``channel_names`` and ``channel_units`` are JSON metadata strings. Voltage
units are recorded as ``V``; other known physical units stay native, and an
unavailable unit is null. Do not treat non-voltage channels as volts.

The default MNE zero-phase FIR filter is 0.5--30 Hz, followed by 100 Hz output.
Set ``l_freq`` and ``h_freq`` to None to disable both bounds, or one to None
for a low-pass or high-pass filter. Cutoffs must be inside the output
Nyquist frequency; MNE also checks the selected source rate. The output rate
must produce an integer number of samples in 30 seconds. No interpolation
of nonfinite input is performed. All returned signals must be finite.

Scored epochs and real time
--------------------------

NSRR stage codes 0, 1, 2, 3/4 and 5 map to W, N1, N2, N3 and R. Codes 6
(movement) and 9 (unscored) are omitted. Non-stage events such as respiratory
or arousal annotations do not define labels. The reader checks the
30-second annotation grid and rejects unknown codes or overlapping stage
intervals. Unannotated gaps remain gaps.

``start_at`` defaults to zero and ``end_at=None`` selects the whole recording.
Both refer to original recording seconds; end_at is exclusive. A partial
crop retains only complete epochs on the original scoring grid. A partial
signal tail is omitted. Sample metadata retains the original start/end time
instead of resetting the first retained epoch to zero. :doc:`sequences`
uses these times to avoid joining movement, unscored or unannotated gaps.

For two channels and 100 Hz output, each sample has shape [2, 3000]::

    from torcheeg import transforms
    from torcheeg.datasets import SHHSDataset

    dataset = SHHSDataset(
        root_path='./SHHS', visits=[1],
        channels=['EEG', 'EOG(L)'], sfreq=100,
        io_path='./cache/shhs-eeg-eog',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Mapping({'Sleep stage W': 0, 'Sleep stage N1': 1,
                                'Sleep stage N2': 2, 'Sleep stage N3': 3,
                                'Sleep stage R': 4})]))

Use a fresh preprocessing and split cache when source files, visits, channel
order, rate, filter or crop settings change. Existing caches are preserved.
This reader does not trim wake periods or standardize a whole night. Record
those choices explicitly when reproducing a published protocol.

Standalone training
-------------------

Run :doc:`../auto_examples/examples_shhs_eegnet` for a two-channel EEGNet
workflow. It keeps the final participant in sorted order for testing,
trains for a fixed epoch count, and does not select settings from that test
participant. Its per-epoch normalization and model differ from a full
SleepDG cross-dataset protocol. Small cropped runs verify the workflow;
they do not establish a benchmark score.
