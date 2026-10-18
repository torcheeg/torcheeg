Train SleepDG across five datasets
=======================================

:doc:`../auto_examples/examples_sleepdg_cross_dataset` combines the native
:class:`torcheeg.models.SleepDG`, :class:`torcheeg.trainers.SleepDGTrainer`
and :class:`torcheeg.datasets.SequenceDataset` with all five sleep readers.
It holds out one complete dataset for testing and uses the other four for
training and source-only checkpoint selection. It supplies no trained weights
or published benchmark score.

Data access and channels
-----------------------------

Obtain each original dataset from its provider. No restricted data are
downloaded by this example. Supply the following roots and exact channel
configuration:

.. list-table:: Inputs at 100 Hz output
   :header-rows: 1

   * - Argument
     - Expected root
     - EEG then EOG
   * - --sleepedf-root
     - Flat Sleep-EDF sleep-cassette EDF/hypnogram pairs
     - EEG Fpz-Cz; EOG horizontal
   * - --hmc-root
     - HMC recordings directory
     - EEG F4-M1; EOG E1-M2
   * - --isruc-root
     - Directory containing Subgroup_1
     - F4-A1 or F4-M1; LOC-A2 or LOC-M2
   * - --shhs-root
     - SHHS directory or polysomnography subdirectory
     - EEG; EOG(L), visit 1
   * - --p2018-root
     - P2018 training directory
     - C3-M2; E1-M2

ISRUC uses subgroup code 0 (Subgroup_1). Candidate names select the actual
recorded reference. Bare-channel differences are computed only when the
required components exist, and the reader records the derivation. They do
not assert that A1 and M1 references are interchangeable. A missing or
ambiguous requested channel fails instead of being filled with zeros.

Read :doc:`sleepedf`, :doc:`hmc`, :doc:`isruc`, :doc:`shhs` and :doc:`p2018`
for pairing, label mappings, actual time and cache requirements. A source
directory may be an explicitly prepared subset, but the example has no
hidden record-count limit. Each training source needs at least two distinct
participants with complete usable sequences.

Preprocessing and explicit prototype differences
-----------------------------------------------------

The author implementation at revision 01d209d4fda874933a277a88bccdca6e3835e884
uses different preparation scripts for different datasets. This example
states the corrected TorchEEG pipeline rather than claiming those scripts
all use the same filter or statistics:

.. list-table:: Reader preprocessing before sequence preparation
   :header-rows: 1

   * - Source
     - Filter and rate order
   * - Sleep-EDF
     - MNE FIR 0.3--35 Hz, then 100 Hz resampling if needed.
   * - HMC
     - No filter; MNE 100 Hz resampling, matching the author's active
       HMC script with its filter line commented out. Whole scored night.
   * - ISRUC
     - MNE FIR 0.3--35 Hz, then 100 Hz resampling.
   * - SHHS
     - MNE FIR 0.3--35 Hz, then 100 Hz resampling.
   * - P2018
     - WFDB FFT to 100 Hz, then eighth-order Butterworth SOS 0.3--35 Hz.

The author's ISRUC/SHHS scripts resample before FIR filtering. Its P2018
script uses direct polynomial Butterworth coefficients and an arousal-based
crop that can misalign stage labels. Its SHHS script reads Profusion XML,
maps unscored code 9 to wake, and slices the first 150 recordings. The
TorchEEG readers keep the corrected stage clock, omit unscored intervals,
use stable SOS coefficients and read the selected source directory in full.
These are declared differences, not an exact published-score reproduction.

Sleep-EDF retains scored sleep plus up to 60 complete wake epochs on each
side. This fixed offline data-selection rule uses the supplied hypnogram;
it is not an online sleep-boundary detector. Non-stage intervals and gaps
remain gaps. Every continuous run is truncated to a multiple of 20 epochs;
no sequence crosses a gap, record, visit, participant or source partition.
Records with no complete sequence are listed in the report. A partition
with none raises an error. An all-wake Sleep-EDF record raises an error
rather than guessing a sleep boundary.

For every retained record, fit a separate population mean and standard
deviation per channel across its retained epochs and time samples. A
constant channel uses scale one. Apply these statistics and return FP32
signals. Streaming statistics use FP64; no record's values enter another
record's statistics. Voltage scaling cancels under this explicit per-channel
normalization: MNE's volts and P2018's native microvolts become dimensionless
standardized values. This does not make their references interchangeable.

Statistics include the unlabeled target recording. This is an explicit
record-wise normalization protocol, not normalization fitted only on
training subjects. The author's scripts normalize at different points
relative to crop, scoring and tail truncation. This example normalizes only
the complete retained epochs, consistently across all five readers.

Partitions, sampling and selection
---------------------------------------

For each source, a seeded permutation holds out ceil(20% of participants),
with at least one, for validation. Repeated visits remain with that person.
The remaining participants form training data. This differs from the author's
random split of already-created source sequences, which can mix one
participant across training and validation.

Assign codes 0--3 to the four source datasets, not to subjects or sequences.
Every training batch contains equal numbers of sequences from all four
sources. Smaller sources cycle through independently shuffled sequences;
an epoch covers the largest source. Batch size must be a multiple of four.
Source validation and target testing use sequential full loaders. Batch
limits are optional, explicit functional-check settings and are recorded.

Use the native width-512, sequence-20, two-channel, 3000-sample architecture.
The complete classification, reconstruction and alignment objective uses
:class:`torcheeg.trainers.SleepDGTrainer` defaults. Adam uses lr=0.001 and
weight decay=0.0001. Norm clipping is 1.0. Cosine scheduling steps after each
successful update, with a recorded horizon. Fifty epochs is the default.
Balanced source batches and participant-held-out validation are explicit
protocol changes relative to the author's sampler.

Only source validation accuracy selects a checkpoint. Target batches do not
enter training, alignment or checkpoint selection. Evaluation restores the
actual selected checkpoint path and runs classification without the decoder.
Target scores from a short subset or a limited loader must not be compared
with complete benchmark results.

Artifacts and trusted resume
---------------------------------

Use a fresh output directory for a new experiment. It contains separate
reader caches, ``run.json`` with data/configuration/partition/statistics,
source-selected and last Lightning checkpoints, ``report.json`` with the
actual evaluated scope, and ``model.pt`` with all native model tensors and
the constructor configuration. Export checks strict state restoration and
exact inference agreement on a target probe.

Resume an owned, trusted end-of-epoch checkpoint with ``--resume PATH``.
The saved checkpoint and run manifest must match the current target, source
roots, partitions, statistics, batch settings, seed and scheduler horizon.
The example restores optimizer/scheduler state through Lightning and starts
the deterministic source sampler at the following epoch. It does not promise
exact mid-epoch recovery. Keep ``--epochs`` unchanged when resuming an
interrupted planned run. To split a check across runs with different epoch
limits, set the same explicit ``--scheduler-steps`` horizon in both runs.
Full Lightning checkpoints contain trusted Python state; they are distinct
from a plain exported tensor-state model.

The JSON records limits, selected epoch/optimizer step, completed training
steps, target metrics and the exact record/participant scope. Report the
scope with any result. No output is added automatically to the leaderboard.
