BCI Competition IV 2b native MAT records
=======================================

:class:`torcheeg.datasets.BCICIV2bDataset` reads BNCI 004-2014's labeled
MATLAB files directly. Download them from the `BNCI dataset page
<https://bnci-horizon-2020.eu/database/data-sets>`_ and cite Leeb et al. and
Tangermann et al. Retain the provider's data terms. Files have names such as
``B01T.mat`` and ``B01E.mat``; they are not the individual-session GDF files
from the original competition distribution.

Each training MAT file contains sessions 1, 2 and 3; the evaluation MAT file
contains sessions 4 and 5. Session structures must have ``X``, ``trial``,
``y``, ``fs`` and ``artifacts``. The reader validates finite six-channel
signals at 250 Hz, ordered complete trial windows and aligned labels/flags.
The published MAT release includes evaluation labels; an unlabelled GDF or
missing label vector is not silently assigned a target.

The first three signals are bipolar C3, Cz and C4 EEG; the last three are
EOG, named EOG1/EOG2/EOG3 here in source order without inferring electrode
positions. ``num_channel=3`` is the default. Explicitly selecting up to six
channels supports artifact analysis. The `competition description
<https://www.bbci.de/competition/iv/desc_2b.pdf>`_ excludes EOG as classification
features. Use the three EEG channels for a competition-style classifier.
The EEG grid, adjacency and region constants use uppercase ``CZ`` in
``BCICIV2B_CHANNEL_LIST`` and retain this same three-channel order. The grid
is a display layout; standard distances are generic, not an individual
participant's bipolar electrode geometry.

Select a time window explicitly
--------------------------------

.. code-block:: python

   from torcheeg import transforms
   from torcheeg.datasets import BCICIV2bDataset

   dataset = BCICIV2bDataset(
       root_path='./BCICIV_2b_mat', io_path='./cache/bciciv2b-native',
       num_channel=3, sfreq=128, offset_start=750, offset_end=1750,
       chunk_size=-1, l_freq=4., h_freq=38.,
       online_transform=transforms.ToTensor(),
       label_transform=transforms.Compose([
           transforms.Select('label'), transforms.Lambda(lambda label: label - 1)]))
   eeg, label = dataset[0]  # [3, 512], left=0 or right=1

MAT ``trial`` indices are one-based. The reader subtracts one exactly once.
Offsets are always source sample counts at 250 Hz, even when ``sfreq``
changes. The default 375:1500 retains the earlier draft's 1.5--6 second
window; it is not a claim to be the complete imagery interval. The explicit
750:1750 selection above uses 3--7 seconds. Both bounds are converted to the
nearest output sample grid. Nonpositive ``chunk_size`` uses the full selected
window, giving 1125 points for the default at 250 Hz or 576 at 128 Hz.
Positive chunk sizes and overlap are output sample counts; incomplete tails
are omitted. Invalid or nonadvancing windows raise an error before cache IO.

Processing runs on a whole continuous MAT session before trial extraction:
optional notch, optional MNE FIR bandpass, optional average EEG reference,
then MNE resampling. Explicit selected EOG channels are filtered and resampled
but are not EEG-referenced. Signals are converted to volts only inside MNE;
reader outputs remain native microvolts. With no processing at 250 Hz, source
values are preserved exactly. Filters are configured against the source
Nyquist frequency. No normalization, amplitude clipping or artifact repair
is implicit. The reader does not globally suppress warnings.

Metadata and partitions
-----------------------

Labels remain 1 (left) and 2 (right); map them explicitly for a classifier.
``artifact`` retains expert flags 0/1. ``skip_trial_with_artifacts=False``
keeps all trials; True explicitly removes flagged trials. This choice must
be recorded in an evaluation protocol. It does not by itself reproduce the
competition's continuous prediction and artifact-free kappa scoring.

``subject_id`` is 1--9. ``session_id`` is 1--5 across T/E files; ``phase`` is
``train`` or ``test``. Record and trial IDs include phase/session, so different
files do not collide. ``trial_index`` identifies a trial within its original
session. ``source_trial_onset`` is its zero-based source sample location.
``start_at``/``end_at`` are output sample indices relative to that resampled
trial origin, with end exclusive. ``source_start_seconds`` and
``source_end_seconds`` give actual output-grid times in the original session.
Channel names, output rate and the native ``uV`` unit are stored with each row.

Keep all windows of a trial in one partition. Subject holdout keeps all five
sessions together; a phase split instead uses the released T/E division.
The native entry does not choose one of these protocols for you. The existing
:doc:`moabb` entry and its independent example remain available. Its paradigm
and native MAT settings must be matched before comparing results.

Hooks receive independent selected-window and chunk copies. ``before_trial``
must preserve finite window geometry. Use new preprocessing and split caches
after adopting this reader or changing data, filters, rates, windows, reference
or artifact policy. Earlier draft caches used the wrong MATLAB origin,
resampled with unchanged crop indices and repeated T/E session/trial IDs.
Preserve those caches for their original runs; do not relabel rows in place.
