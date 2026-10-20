ADSZ Alzheimer's disease and schizophrenia EEG
===============================================

Sources and separate tasks
----------------------------

:class:`torcheeg.datasets.ADSZDataset` reads the two cohorts in
`Figshare article 19091771 version 1
<https://doi.org/10.6084/m9.figshare.19091771.v1>`_, released under CC0.
Cite Alves et al., *EEG functional connectivity and deep learning for automatic
diagnosis of brain disorders: Alzheimer's disease and schizophrenia*,
Journal of Physics: Complexity 3(2), 025001 (2022),
https://doi.org/10.1088/2632-072X/ac5f8d.

The fixed ``dataset.zip`` is Figshare file ``33928037``, 18,277,631 bytes,
with published MD5 ``61d6cfb2f6a31802fde5229bb5cf86d5`` and verified SHA256
``bfa0306a0fd6b3e4e90dc286d302a5515c18245a4c4d9a57f67391d6677d3fc1``.
It contains 48 Alzheimer/healthy-control ``.out`` records (24 per class)
and 84 schizophrenia/healthy-control ``.eea`` records (45/39). These are two
independent binary cohorts; their controls are not one shared clinical group.
Use ``subset='alzheimer'`` (default) or ``subset='schizophrenia'``.

The schizophrenia cohort was obtained and prepared by N. N. Gorbachevskaya
and S. V. Borisov. The `original MSU description
<http://brain.bio.msu.ru/eeg_schizophrenia.htm>`_ supplies its file layout,
channel order, microvolt units and one-file-per-subject rule. This release
contains adolescents with symptoms of schizophrenia. Its label is not an
assertion of a separately verified clinical diagnosis for each person.

Point ``root_path`` at the extraction root, its ``dataset`` directory or the
selected cohort directory. More than one matching root is an error.
The two complete class directories are required; missing/empty classes,
unrecognized source filenames and duplicate case-insensitive record IDs fail.
The reader does not replace one cohort with the other.

.. code-block:: text

    ADSZ/
      dataset/
        alzheimer/
          AD/ec0101.out
          AD/eo0101.out
          Healthy/eeg20c1.out
          Healthy/eeg20o1.out
        schizophrenia/
          SCH/022w1.eea
          Healthy/S10W1.eea

Channels, layout and units
---------------------------

Alzheimer matrices have one sample per row and 19 channels per row, in
``ADSZ_CHANNEL_LIST`` order:
``Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2``.
Keep all source rows. The fixed archive has lengths from 1,024 to 1,792
samples, so do not truncate every record to eight seconds. The source rate
is 128 Hz. The archive does not supply a physical unit table for these files;
the default preserves their numbers with ``signal_unit='source'`` and
``units_basis='unspecified_source_unit'``.

Schizophrenia files contain a single column of 122,880 numbers: the first
7,680 values are channel 1, the next 7,680 channel 2, and so on. This gives
16 channels by 60 seconds at 128 Hz, not interleaved samples. The documented
``ADSZ_SCHIZOPHRENIA_CHANNEL_LIST`` order is
``F7, F3, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2``.
The provider specifies microvolts, so this subset converts numbers to volts
and records ``units_basis='provider_microvolts'``.

``source_unit`` explicitly overrides the interpretation: ``'source'`` keeps
numbers without a physical claim, ``'uV'`` converts to volts, and ``'V'``
keeps numbers as caller-declared volts. Overrides have
``units_basis='caller_supplied_unit'``; they do not verify hardware calibration.
The original unreviewed AD addition unconditionally assumed microvolts.
Set ``source_unit='uV'`` to reproduce that scaling, after establishing the
source convention for your data. Use a fresh cache when changing this setting.

The native defaults add no notch, filter, reference or resampling. Optional
processing runs on all recorded channels in this order: notch, FIR,
average reference, then resampling. Select the first ``num_channel`` channels
afterward. None selects all native channels (19 or 16). Cutoffs must be below
the 128 Hz source Nyquist; the caller remains responsible for an appropriate
band limit when reducing the sampling rate. Linear processing retains the
chosen numerical unit convention.

Windows and evaluation identities
----------------------------------

The default uses nonoverlapping 128-sample windows. Nonpositive
``chunk_size`` returns the complete record. Omit incomplete trailing windows.
Overlap must be nonnegative and smaller than the actual window. Window and
hook inputs are independent copies; a before-trial hook must preserve finite
geometry. Source bounds refer to the corresponding stored recording in
seconds; they do not reconstruct any unreleased acquisition clock.

Alzheimer filenames are not a clinical identity table. Conservatively group
AD files with equal digits after ``ec``/``eo``, and Healthy files with equal
``eeg<number>`` prefixes across ``c/o``. For example, ``ec0101`` and ``eo0101``
share one group; ``eeg20c1`` and ``eeg20o1`` share another. Keep all 48 records;
the fixed release yields 24 such conservative groups. This grouping can merge
records that a clinical identity table would distinguish. Metadata mark
``identity_basis='conservative_filename_group_not_clinically_verified'``.
Do not infer a verified eye condition or number of clinical participants
from these codes. The paper's reported subject counts are not a filename
linkage specification.

The MSU description states that each file represents one subject. Keep the
complete name as its grouping key; do not merge different files just because
their names share a leading number. Each subset and diagnosis has a separate
ID namespace. Labels are HC 0 / AD 1 or HC 0 / SCH 1. ``record_id`` and
``trial_id`` retain file identity; ``subject_id`` retains the grouping rule.
Keep these groups together in evaluation, and use an external verified clinical
identity table when a stronger unseen-person claim is required. Cached
``after_subject`` hooks use these same groups.

Load native signals
---------------------

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import ADSZDataset

    dataset = ADSZDataset(
        root_path='./ADSZ',
        subset='schizophrenia',
        io_path='./cache/adsz-schizophrenia-native',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'),
    )
    eeg, label = dataset[0]  # [16, 128], volts

Use separate cache roots for the two subsets, source unit choices, source-file
changes and all preprocessing settings; see :doc:`cache`. The Alzheimer
constants preserve the original proposed channel/grid/graph values; the
schizophrenia graph is the induced template on its 16 source electrodes.
These templates are not measured patient connectivity. Source-loading checks
do not establish clinical classification performance. Standalone training
workflows require their own fixed grouping protocol and actual execution.
