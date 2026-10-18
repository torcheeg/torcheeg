ISRUC recordings, references and scorers
========================================

:class:`torcheeg.datasets.ISRUCDataset` reads the original
`ISRUC-Sleep recordings <https://sleeptight.isr.uc.pt/>`_ and one selected
expert's text scores. The three subgroups contain 100 single-night subjects,
eight two-night subjects and ten single-night healthy subjects. ``groups``
uses the existing codes 0, 1 and 2 for these subgroups. ``scorer=1`` selects
``<record>_1.txt``; ``scorer=2`` selects the other expert independently.

The original ``.rec`` files contain EDF data. The reader makes an owned
portable temporary EDF copy, closes it after extracting the signals and
removes only that temporary copy. It does not require Linux ``/dev/shm`` or
modify the recording. Missing and ambiguous file pairs raise errors.

Measured and computed channel references
----------------------------------------

Recordings have different channel inventories. Default candidates select one
of F3-A2/F3-M2, C3-A2/C3-M2, O1-A2/O1-M2, F4-A1/F4-M1, C4-A1/C4-M1 and
O2-A1/O2-M1, in that order. A candidate tuple selects the one available named
channel. Multiple available candidates are ambiguous and require an explicit
name. A1/A2 and M1/M2 are not renamed or asserted to be physically identical.
Inspect the actual recording references for the intended protocol.

Some recordings instead contain separate electrode and reference signals.
With ``derive_channels=True``, a missing name such as C3-A2 is calculated as
recorded C3 minus recorded A2. An existing candidate takes priority over
calculating a different reference. Disable derivation to require an existing
named signal. This option performs real arithmetic, with no synthetic renaming.

For explicit EEG/EOG candidates::

    from torcheeg.datasets import ISRUCDataset
    from torcheeg import transforms

    dataset = ISRUCDataset(
        root_path='./ISRUC-SLEEP', io_path='./cache/isruc-raw-v2',
        groups=[0, 1, 2], scorer=1,
        channels=[('C3-A2', 'C3-M2'), ('LOC-A2', 'E1-M2')],
        missing_channels='raise', derive_channels=True,
        l_freq=0.3, h_freq=35., sfreq=100,
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Mapping({'Sleep stage W': 0, 'Sleep stage N1': 1,
                                'Sleep stage N2': 2, 'Sleep stage N3': 3,
                                'Sleep stage R': 4})]))

MNE reads physical signal values in volts. Filtering and resampling produce
30-second epochs; the example has shape [2, 3000]. It does not normalize
records, trim wake intervals or train a model. The historical explicit missing
channel policy remains ``'zero'``; ``'raise'`` rejects absent signals.
Zero-filled default slots use the first candidate name and are identified as
zero-filled metadata, not as measured signals. Incomplete channel inventories
are not repaired by substituting a different electrode position.

The metadata fields ``channel_names``, ``derived_channels`` and
``zero_filled_channels`` are JSON strings. They report ordered source channel
names, computed channel/component pairs, and zero-filled names. Offline and
online transforms may subsequently change the sample representation; these
fields describe the reader's signal channels.

Stage alignment and sequence boundaries
---------------------------------------

Raw score codes are 0/W, 1/N1, 2/N2, 3/N3 and 5/R. Each selected expert must
supply one valid score per complete 30-second recording epoch. Empty scores,
unknown codes and count mismatches raise an error, rather than silently
truncating one side or guessing where extra labels belong. The reader does
not combine the two experts into a consensus.

Epoch events are exactly 30 seconds apart, with an inclusive final sample at
``30 - 1/sfreq`` within each epoch. A partial recording tail is omitted.
MNE-rejected epochs keep their matching labels through the original event
selection. ``start_at`` and ``end_at`` metadata are seconds from the original
recording start and preserve rejection gaps.

``subject_id`` identifies subgroup and released subject folder. Both nights
of a Subgroup_2 subject share it. ``session_id`` is one or two; ``record_id``,
``trial_id`` and path-free ``clip_id`` distinguish the nights.
``group_id`` keeps codes 0/1/2 and ``scorer_id`` records the selected expert.
Keep both nights of a held-out person together. Use :doc:`sequences` only
after selecting partitions, with no sequence across a record or time gap.

Use a new cache path after upgrading. Previous caches can contain overlapping
epochs from a too-short event interval, silent zero-filled reference names,
night-colliding clip identifiers and fabricated contiguous rejection times.
They remain untouched. Also use new preprocessing and split caches after
changing channels, derivations, scorer, sampling or subject membership.

This reader accepts original REC/text pairs. The separately published
`pre-extracted MATLAB files <https://sleeptight.isr.uc.pt/?page_id=76>`_ omit
30 final epochs according to their provider; that separate preparation is
not silently applied to these raw recordings.
