ADFSU resting-state EEG
=======================

Source release
---------------

:class:`torcheeg.datasets.ADFSUDataset` reads the text arrays from
`Vicchietti et al. (2023) <https://www.nature.com/articles/s41598-023-32664-8>`_
and the corresponding `OSF archive <https://osf.io/2v5md/>`_.
The file ``EEG_data.zip`` is OSF file ``heaw6``, version 1, with SHA256
``f5b30df4fd0d18e3224dde0bd564e2a5cea61845ae5a9b8142ae722c5d99ba93``.
Extract it without renaming the group, condition, subject or channel folders::

    EEG_data/
        Healthy/Eyes_closed/Paciente1/Fp1.txt
        Healthy/Eyes_open/Paciente1/Fp1.txt
        AD/Eyes_closed/Paciente1/Fp1.txt
        AD/Eyes_open/Paciente1/Fp1.txt

Each complete condition has 19 canonical electrode files, each containing
1024 finite numbers: eight seconds at 128 Hz. The fixed channel order is
``Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2``.
The release also includes ``F1.txt`` and ``F2.txt`` in 182 directories;
these are byte-identical copies of ``Fp1.txt`` and ``Fp2.txt``. They are not
additional electrodes and are not loaded.

The archive has 12 Healthy and 80 AD directory names in each condition.
``Healthy/Eyes_open/Paciente5`` is empty. The reader warns and omits completely
empty condition directories, keeping any other complete condition for that
folder identity. A partially missing 19-channel record raises an error, even
if ``num_channel`` selects fewer output channels. Wrong lengths, nonfinite
values and an empty overall selection also raise errors. The published
archive therefore supplies 183 complete condition records, not 184 signals.

The source paper describes 24 healthy and 160 probable AD patients across
its four groups. The archive does not supply a participant linkage table
across the two conditions. The reader uses ``group_PacienteN`` as a
**conservative grouping key**, so equal folder names in the same diagnosis
stay together across conditions. This key does not establish a verified
clinical identity or a count of distinct people. Do not report a verified
cross-condition, unseen-person evaluation from folder names alone. A workflow
limited to one condition avoids relying on a cross-condition pairing table.

Values and optional preprocessing
----------------------------------

By default the reader returns the original text numbers, with no extra
filtering, reference, resampling or amplitude scaling. The archive and its
OSF description do not provide a voltage calibration. Metadata therefore
use ``signal_unit='source'``. Do not label these values as volts or microvolts
without independent calibration information.

The paper describes provider band limiting to 0.5--30 Hz and movement artifact
removal by an EEG technician. These steps are already part of the released
signal. Additional ``notch_freqs``, ``l_freq``, ``h_freq``, ``ref_channels`` and
``sfreq`` settings are explicit options. They do not reproduce the paper's
wavelet feature analysis or infer a physical signal unit.

Additional notch, band filtering and average reference operate on the full
19-channel, eight-second condition, followed by optional resampling. Only
then are the first ``num_channel`` channels selected. These operations are
linear and preserve the source numerical scale. MNE's internal raw array is
used for these operations only; its SI convention is not a source calibration.
Long filters on these short records can produce MNE edge-effect warnings;
choose settings suitable for eight-second inputs rather than suppressing
warnings. The reader does not change global warning filters or MNE logging.

Load windows
-------------

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import ADFSUDataset

    dataset = ADFSUDataset(
        root_path='./EEG_data',
        states=('Eyes_closed',),
        chunk_size=128,
        overlap=0,
        io_path='./cache/adfsu-eyes-closed-native',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'),
    )
    eeg, label = dataset[0]  # [19, 128]; Healthy=0, AD=1

``states`` selects unique ``Eyes_closed`` and/or ``Eyes_open`` names.
A nonpositive ``chunk_size`` returns the complete condition. Window length
and overlap count output samples, including after resampling; incomplete
last windows are omitted. Overlap must be smaller than each window.

Metadata retain the diagnosis, source folder, conservative ``subject_id``,
condition ``session_id`` and a condition-qualified ``record_id``/``trial_id``.
IDs remain stable when selecting just one condition. ``start_at``/``end_at``
are half-open output-sample bounds; ``source_start_seconds`` and
``source_end_seconds`` give the corresponding interval in seconds, starting
at the beginning of this released eight-second segment. No clock within an
unreleased longer recording is inferred. ``channel_names`` is a JSON list.

Hooks and overlapping windows receive independent arrays. A ``before_trial``
hook must preserve finite channel/time geometry. Use a separate cache path
when source files or preprocessing settings change; see :doc:`cache`.
The grid and adjacency constants use the same 19-channel order. The graph
is a hand-defined template, not measured connectivity.

Keep all windows from a condition together in evaluation partitions, and
keep equal conservative grouping keys together if using both conditions.
Never select filters, model settings or epochs with the test labels.
The reader checks signal loading and metadata; it does not establish clinical
prediction accuracy or reproduce a published classifier score.
