Consecutive epoch sequences
===========================

:class:`torcheeg.datasets.SequenceDataset` groups existing epochs into complete
sequences. It reads each source sample through its normal signal and label
transforms. It does not load a raw recording, filter, normalize or relabel it.
Use a reader that preserves real epoch times, such as the corrected
:doc:`sleepedf` or :doc:`hmc` reader, with a fresh preprocessing cache.

Partition before sequence construction
--------------------------------------

Select train, validation and test subjects or records before constructing the
wrappers. Pass each partition separately. If a metadata table already includes
multiple partitions, add its column to ``groupby``. The wrapper always retains
all available dataset-source, subject, session, recording and trial boundaries.
Extra grouping columns cannot remove those boundaries.

For one already partitioned source dataset::

    from torch.utils.data import DataLoader
    from torcheeg.datasets import SequenceDataset

    train_sequences = SequenceDataset(source_train, sequence_length=20,
                                      domain_id=0)
    train_loader = DataLoader(train_sequences, batch_size=4, shuffle=True)
    eeg, labels, domains = next(iter(train_loader))
    # Numeric source samples: eeg [B, 20, C, T], labels [B, 20], domains [B].

The code assumes that each source label transform returns one integer target.
Dictionary labels are instead collated as dictionaries of per-epoch values.
The signal dtype is preserved when numeric arrays become PyTorch tensors.
The final loader batch can be smaller than the configured batch size.

Source metadata needs ``clip_id``, ``start_at`` and ``end_at``. Times are finite
nonnegative seconds from the original recording start; each duration must equal
``epoch_duration`` (30 seconds by default). A ``record_id`` or ``_record_id``
column identifies recordings. For another metadata schema, supply explicit
recording columns in ``groupby``. Missing group values raise an error rather
than silently dropping those epochs. Released record identifiers do not imply
an undocumented cross-night person linkage.

Runs, gaps and tails
--------------------

Rows are sorted by start time inside each group. Duplicate or overlapping
epoch intervals are rejected. A gap begins a new continuous run, and a repeated
source clip in one recording is rejected even if extra partition columns differ.
A window cannot bridge an unscored, rejected or omitted epoch. Floating time
comparison uses an absolute tolerance of at most one microsecond, without a
relative tolerance that increases with recording time.

By default, ``stride=None`` means ``stride=sequence_length``: complete windows
do not overlap. Incomplete tails are dropped independently for each run.
For example, a seven-epoch run with length three returns epochs 0--2 and 3--5;
epoch six is unused. With stride two, it instead returns 0--2, 2--4 and 4--6.
No epoch is padded, repeated or borrowed from another run to fill a tail.
An input with no complete window produces an empty wrapper.

Use ``get_epoch_indices(index)`` to inspect a sequence's positions in the
snapshotted source dataset. ``read_info(index)`` returns group fields,
``sequence_id``, ``clip_id`` and the original sequence start/end seconds.
The summary table does not replace per-epoch labels with one sequence label.
For overlapping windows, randomly splitting sequences afterwards can place
the same epoch in training and test. The wrapper cannot prevent leakage
between separately supplied inputs; check partition IDs and source membership.

Source-dataset codes
-------------------

``domain_id`` adds a constant nonnegative int64 source-dataset code as the
third sample item. Create one wrapper per source, then combine wrappers with
PyTorch's ``ConcatDataset`` if needed. Keep a fixed name-to-code mapping for
all training data. Domain generalization requires source-dataset membership;
subject numbers are not interchangeable with dataset codes.

Alternatively, use ``domain_column`` for integer codes in source metadata.
That column becomes a group boundary. It is mutually exclusive with
``domain_id``. The returned code comes from the bound source, so a CSV view's
summary-field conversion cannot silently change the sequence's domain.
Without either option, samples contain only ``(eeg, labels)`` for evaluation
or ordinary sequence classification.

Copies, workers and saved partitions
------------------------------------

The wrapper copies the source dataset and snapshots its metadata. Signal
storage and transform objects remain shared, as in normal TorchEEG views.
Do not mutate shared storage, transforms or nested source datasets while
loading a wrapper. Ordinary source metadata edits do not reorder its epochs.
Pickle serialization and spawned-worker DataLoaders preserve epoch order.

Sequence IDs bind source epoch identities, their metadata order and the window
configuration. Shallow copies and CSV-backed TorchEEG partition views retain
those IDs and resolve the original windows. Changing the source order,
sequence length, stride, domain setting or grouping invalidates old IDs;
use a new ``split_path``. The wrapper raises on an unknown sequence ID instead
of treating a cached row number as a new source position. This check does not
hash signal files or replace the source reader's cache validation.
