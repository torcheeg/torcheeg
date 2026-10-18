BCI Competition IV 2a MAT data
==============================

``BCICIV2aDataset`` reads the BNCI MATLAB release of dataset 001-2014.
Download ``A01T.mat`` through ``A09E.mat`` from the
`BNCI source <https://bnci-horizon-2020.eu/database/data-sets>`_.
The reader requires named ``X``, ``trial``, ``y``, ``fs`` and ``artifacts``
fields. It does not read the original GDF files or recover hidden labels.

Signals and identities
----------------------

The first 22 columns are EEG; the last three are EOG. Native signals remain
in microvolts at 250 Hz. The default selects 22 EEG channels and one
1750-sample window per trial. Labels remain 1 (left hand), 2 (right hand),
3 (feet) and 4 (tongue). Map them to 0--3 for cross-entropy training.

Each file contains continuous runs. Calibration runs have no motor-imagery
trials and yield no samples. Their positions remain in the zero-based
``run`` metadata. A04T has a shorter calibration block. The reader uses each
run's fields rather than assuming a fixed number of calibration blocks.

``subject_id`` and ``subject`` retain strings such as ``A01``. ``session``
retains ``T`` or ``E``; ``phase`` adds ``train`` or ``test``. ``trial_index``
is the zero-based trial within a run. ``trial_id`` qualifies this index by
file and run, so a trial-based splitter cannot mix different runs under one
ID. Overlapping windows of one trial retain the same trial ID. ``artifact``
retains the provider's flag. ``skip_trial_with_artifacts=True`` excludes
flagged trials without renumbering their source trial indices.

Time windows and preprocessing
------------------------------

MAT trial positions are one-based. The reader subtracts one exactly once.
``offset`` remains the source start offset at 250 Hz. The new
``offset_start`` is an explicit alias; if both are supplied, their nonzero
values must agree. ``offset_end`` is an exclusive source offset, with a
default of 1750. All selected windows stay within the first seven seconds
of a trial, including the released reader's default 0--7 second selection.
No sample extends into the next trial or the intervening rest period.

Optional notch filtering, FIR filtering, EEG average reference and
resampling operate on each continuous motor-imagery run before extracting
trials. MNE receives volts internally; the output returns to microvolts.
EOG channels retain their channel type and do not enter the EEG reference.
They are available for artifact analysis and are not classifier inputs in
the original competition protocol.

Offsets always count **source** samples. ``chunk_size`` and ``overlap`` count
**output** samples. Nonpositive chunk sizes select the complete chosen
window. Positive chunk sizes must fit it. For example, 2--6 seconds at
128 Hz produces one 512-sample window:

.. code-block:: python

   from torcheeg import transforms
   from torcheeg.datasets import BCICIV2aDataset

   dataset = BCICIV2aDataset(
       root_path='./BCICIV_2a_mat',
       sfreq=128,
       l_freq=4.0,
       h_freq=38.0,
       offset_start=500,
       offset_end=1500,
       chunk_size=-1,
       io_path='./cache/bciciv2a-128hz-2to6s',
       online_transform=transforms.Compose([
           transforms.To2d(), transforms.ToTensor()]),
       label_transform=transforms.Compose([
           transforms.Select('label'),
           transforms.Mapping({1: 0, 2: 1, 3: 2, 4: 3})]))

``source_trial_onset`` records the zero-based onset within its source run.
``start_at`` and ``end_at`` are absolute sample positions in the processed
run, using the rounded resampled trial origin. ``source_start_seconds`` and
``source_end_seconds`` express those actual output-grid positions in seconds.
``channel_names``, ``sfreq`` and ``signal_unit`` record the returned geometry.
For compatibility, ``before_trial`` still operates once on the complete
motor-imagery run, retaining all 25 input channels before selecting
``num_channel`` output channels. It must preserve its shape and finite values. Each window
is copied before an offline transform, including overlapping windows.

Evaluation and migration
------------------------

For within-person evaluation, train on that person's T file and evaluate
on their E file. Keep evaluation labels out of model selection. Other
protocols need explicit subject, session and trial isolation.
The optional MNE FIR and FFT processing is an offline protocol. It does not
implement the original competition's causal, continuous-output, artifact-free
kappa evaluation. The cue starts at two seconds; imagery ends at six seconds.
See the `official description <https://www.bbci.de/competition/iv/desc_2a.pdf>`_.

The existing constructor's positional order, ``offset``, artifact policy,
default 22-channel/1750-sample geometry and subject/session fields remain.
The reader fixes the old one-sample onset shift, repeated trial IDs across
runs and chunks that could extend into rest. It rejects invalid or incomplete
trials rather than silently dropping them. A positive chunk that no longer
fits after an offset or resampling raises an error; use ``chunk_size=-1``
or an explicit output size. These correctness fixes change cached values
and trial metadata. Use a **new cache path** when adopting this reader.
Do not delete an earlier experiment's cache to migrate.

Grid, graph and region constants
-------------------------------

The legacy ``BCICIV2A_CHANNEL_LIST``, ``BCICIV2A_LOCATION_LIST`` and
``BCICIV2A_LOCATION_DICT`` retain their original aliases and grid. For example,
``'0'`` labels the second signal column. It is not a standard electrode name.
``BCICIV2A_STANDARD_CHANNEL_LIST`` gives the physical EEG names in the same
22-column order. Position zero is FZ, alias 0 maps to FC3, alias 2 maps to FCZ,
and alias 16 maps to POZ. Names use uppercase as in the other library constants.
The order agrees with the BNCI loader in
`MOABB <https://github.com/NeuroTechX/moabb/blob/0d173bf5a745cc9e861f48c3ee41f1e9d336eb6a/moabb/datasets/bnci/bnci_2014.py>`_.
EOG channels are not included in these 22-node EEG constants.

``BCICIV2A_ADJACENCY_LIST`` and ``BCICIV2A_ADJACENCY_MATRIX`` connect occupied
horizontal, vertical and diagonal neighboring grid cells. The matrix is binary,
symmetric and has no self-loops. Gaps are not bridged and grid boundaries do
not wrap. The reusable
:func:`~torcheeg.datasets.constants.utils.format_adj_list_from_location_list`
constructs this eight-neighbor list from rectangular grids, with ``'-'`` as
the empty-cell marker. It rejects repeated names, ragged grids and invalid cells.

``BCICIV2A_STANDARD_ADJACENCY_MATRIX`` uses the physical names with the library's
standard 10--20 coordinate template and ``delta=0.00035``. It retains the
existing inverse-distance weighting and directed global-link correction:
C5 to C6 subtracts one from that weight; C6 to C5 retains the distance weight.
The matrix therefore has a negative directed edge and diagonal values of one.
This is a model prior, not measured subject coordinates or functional connectivity.
Do not assume that this matrix is binary, symmetric or free of self-loops.

``BCICIV2A_GENERAL_REGION_LIST``, ``BCICIV2A_FRONTAL_REGION_LIST`` and
``BCICIV2A_HEMISPHERE_REGION_LIST`` group physical names and return indices
into the original signal-column order. Each grouping covers all 22 EEG channels.
Passing the numeric aliases directly to standard-name coordinate or region
lookups would leave most channels unassigned.

With the graph dependencies installed, the following constructs both
representations without changing channel order. The array is synthetic usage
input and does not measure a training score:

.. code-block:: python

   import numpy as np
   from torcheeg import transforms
   from torcheeg.datasets.constants import (
       BCICIV2A_LOCATION_DICT, BCICIV2A_STANDARD_ADJACENCY_MATRIX)

   eeg = np.arange(22 * 128, dtype=float).reshape(22, 128)
   to_grid = transforms.ToGrid(BCICIV2A_LOCATION_DICT)
   grid = to_grid(eeg=eeg)['eeg']
   recovered = to_grid.reverse(eeg=grid)['eeg']
   graph = transforms.ToG(
       BCICIV2A_STANDARD_ADJACENCY_MATRIX,
       add_self_loop=False)(eeg=eeg)['eeg']

``grid`` has shape ``[128, 6, 7]`` and ``recovered`` retains the original EEG
values. ``graph.x`` has shape ``[22, 128]``; its node order matches the source
columns. ``add_self_loop=False`` keeps the standard matrix's existing diagonal
weights rather than adding another identity matrix. Keep the graph and region
constants aligned with the selected EEG columns if you select fewer channels.
