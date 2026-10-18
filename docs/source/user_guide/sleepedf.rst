Sleep-EDF records and epoch metadata
====================================

:class:`torcheeg.datasets.SleepEDFxDataset` reads the expanded Sleep-EDF
cassette and telemetry studies. Download the EDF files from the
`PhysioNet dataset page <https://www.physionet.org/content/sleep-edfx/1.0.0/>`_
and retain its study subdirectories.

.. code-block:: text

   sleep-edf-database-expanded-1.0.0/
   ├── sleep-cassette/
   │   ├── SC4001E0-PSG.edf
   │   └── SC4001EC-Hypnogram.edf
   └── sleep-telemetry/
       ├── ST7021J0-PSG.edf
       └── ST7021JM-Hypnogram.edf

Files are paired by the first six filename characters: study, subject and
night. The final PSG and hypnogram characters can differ. Missing pairs and
multiple files for one subject/night raise an error. A missing hypnogram
cannot shift the pairing of later recordings.

Load scored epochs
------------------

.. code-block:: python

   from torcheeg import transforms
   from torcheeg.datasets import SleepEDFxDataset

   dataset = SleepEDFxDataset(
       root_path='./sleep-edf-database-expanded-1.0.0',
       studies=['cassette'],
       channels=['EEG Fpz-Cz', 'EOG horizontal'],
       missing_channels='raise',
       l_freq=0.5,
       h_freq=30,
       sfreq=100,
       io_path='./cache/sleepedf-record-metadata',
       online_transform=transforms.ToTensor(),
   )
   eeg, info = dataset[0]

Each signal has shape ``[channels, 30 * sfreq]`` in volts before a user
transform. Channel order follows ``channels``. Whole-record FIR filtering
precedes resampling and epoch extraction. The historical default
``missing_channels='zero'`` fills an absent channel with zeros. Select
``'raise'`` when the model requires measured EEG or EOG channels.

The reader selects complete 30-second scored intervals. It excludes movement,
unscored and unrelated annotations, and drops incomplete end intervals or
epochs rejected by MNE. Event codes distinguish source stages 3 and 4;
metadata then maps both to ``Sleep stage N3``. The other labels are
``Sleep stage W``, ``Sleep stage N1``, ``Sleep stage N2`` and ``Sleep stage R``.
Apply a label transform to obtain integer targets for classification.

Keep subjects and recording boundaries
--------------------------------------

For ``SC4001E0-PSG.edf``, metadata contains:

.. list-table::
   :header-rows: 1

   * - Field
     - Value or unit
   * - ``subject_id``
     - ``SC400``; both nights of this person share the ID
   * - ``session_id``
     - ``1``; the night number
   * - ``record_id`` / ``trial_id``
     - ``SC4001E0``; one complete PSG recording
   * - ``clip_id``
     - ``SC4001E0_0`` for the first retained epoch
   * - ``start_at`` / ``end_at``
     - Seconds from recording start; each interval spans 30 seconds

Use ``subject_id`` for unseen-person partitions. Study prefixes keep cassette
and telemetry subjects distinct. Use ``record_id`` to keep sequences within
one recording. Two retained epochs can have a gap after a movement or
unscored interval: clip indices alone do not establish continuity. Before
building a sequence, check that each next ``start_at`` equals the preceding
``end_at`` and that all record IDs match. The reader itself returns individual
epochs; it does not silently assemble a sequence.

``before_trial`` receives the retained epochs for one recording. It must
preserve their count and order so that labels and timestamps still match.
Direct legacy calls to ``process_record`` without ``epochs_onsets`` assume
contiguous intervals starting at zero. ``read_record`` always supplies actual
onsets; supply them yourself for an external epoch source with gaps.

Keep preprocessing and cache history explicit
---------------------------------------------

Use a fresh ``io_path`` after upgrading this reader. Previous caches can
contain path-based subject/clip identifiers and fabricated contiguous times.
The cache configuration check can reject the old generation; preserve that
cache for the original experiment. Do not relabel old rows without re-reading
the source annotations.

This reader does not trim wake epochs, normalize a recording, perform a
train/validation/test split, or reproduce a specific paper protocol. Select
those steps explicitly. For standard holdout evaluation, fit data-dependent
preprocessing on training data only. State any protocol that instead uses
unlabeled target-record statistics. Local reader validation is separate from
training a complete sleep-stage or domain-generalization method.
