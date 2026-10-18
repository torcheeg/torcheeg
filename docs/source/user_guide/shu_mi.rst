SHU-MI trial records
====================

:class:`torcheeg.datasets.SHUMIDataset` reads the published four-second trial
MAT files from Ma et al. (2022), `Scientific Data
<https://www.nature.com/articles/s41597-022-01647-1>`_. Download the trial
archive from the `SHU Multi-session Dataset
<https://figshare.com/articles/code/shu_dataset/19228725>`_ and extract files
such as ``sub-001_ses-01_task_motorimagery_eeg.mat`` into one directory.
Retain the provider's metadata and cite the paper and dataset.

This entry reads trial MAT files directly without a MOABB dependency. Real-data validation uses
the version 1 trial archive; retain the source version in experiment records. MOABB's
Ma2022 dataset instead provides a continuous EDF/BIDS workflow. The two paths
have different input and event handling; their outputs are not automatically
interchangeable protocols.

The provider has already removed bad segments and baselines and applied a
0.5--40 Hz FIR bandpass. Each MAT file contains ``data`` with shape
``[retained_trials, 32, 1000]`` at 250 Hz and ``labels`` containing 1 (left)
or 2 (right). Some sessions have fewer than 100 retained trials. Row, column
and vector labels are accepted when their count matches. Invalid labels,
nonfinite signals or a different trial geometry raise an error.

Load trial windows
------------------

.. code-block:: python

   from torcheeg import transforms
   from torcheeg.datasets import SHUMIDataset

   dataset = SHUMIDataset(
       root_path='./SHU-MI', io_path='./cache/shu-mi-trials',
       sfreq=200, chunk_size=200, overlap=0, num_channel=32,
       online_transform=transforms.ToTensor(),
       label_transform=transforms.Select('label'))
   eeg, label = dataset[0]  # [32, 200], left=0 or right=1

The reader uses SciPy FFT resampling along each trial's time axis, with
float64 arithmetic to retain precision near large source transients. It does
not add a filter, normalize amplitudes, divide by 100, or convert microvolts
to volts. A rate of 250 Hz keeps the source values unchanged. All returned
values remain in microvolts until an explicit user transform changes them.
Labels are read from the MAT array itself; an events TSV is not substituted.
All published channels are retained unless explicitly selected; this reader
does not repair bad channels or repeat the provider's artifact rejection.

Channels follow ``SHU_MI_CHANNEL_LIST`` in ``torcheeg.datasets.constants``.
``num_channel`` selects its first channels, not a named montage. The grid and
adjacency constants follow this same order. The grid is a display layout;
the standard-distance adjacency uses generic electrode coordinates, not
measured positions from an individual participant.

Set ``chunk_size=-1`` (or zero) to return each full resampled four-second
trial, for example [32, 800] at 200 Hz. Positive chunks omit an incomplete
tail. Overlap must be nonnegative and smaller than the effective chunk size.
An invalid step is rejected before reading or creating a cache.

Metadata and evaluation
-----------------------

``subject_id`` and ``session_id`` retain the filename's ``sub-XXX`` and
``ses-YY`` forms. ``record_id`` identifies that session file. ``trial_index``
is the zero-based position among retained MAT trials; ``trial_id`` includes
the complete record ID, so trials from different sessions remain distinct.
``start_at`` and ``end_at`` are output sample indices within a trial, with an
exclusive end. Divide by the stored ``sfreq`` to obtain trial-relative
seconds. They do not reconstruct the original continuous recording clock.
The cache records the selected channel names and ``signal_unit='uV'``.

Keep every chunk of a trial in one partition. For an unseen-person protocol,
partition on ``subject_id`` so all sessions stay together. For cross-session
validation, separate the chosen sessions explicitly and describe whether the
participant is shared. The reader does not choose an evaluation protocol or
establish a published benchmark score.

``before_trial`` receives an independent selected-channel trial copy and must
preserve its shape, finite values and sample order. Offline transforms then
receive independent chunk copies, including when windows overlap. Existing
session/subject postprocessing hooks use the normal BaseDataset interface.

Use fresh preprocessing and split caches when adopting this reader or
changing inputs, channels, sample rate, chunks or transforms. Earlier draft
caches used session-local integer trial IDs and lacked unit provenance.
Preserve those caches for the original experiments. A separate standalone
training example and real-data benchmark results require their own validation;
rendering this guide does not execute training.

Train a standalone EEGNet
-------------------------

:doc:`/auto_examples/examples_shu_mi_eegnet` runs an explicit participant
holdout with the native trial reader. Install the source with ``dataset``
and ``trainer`` extras and run:

.. code-block:: console

   python examples/examples_shu_mi_eegnet.py --root-path ./SHU-MI \
       --epochs 1 --output-path ./shu-mi-eegnet.pt

Every model input is one complete four-second trial: 32 published channels
resampled to 200 Hz, shape [1,32,800], then normalized per trial/channel.
No extra filter, division by 100, bad-channel repair or target-data fitting
is introduced. The provider's preprocessing is recorded in the model report.

The last sorted participant is held out with every available session; all
others train for a fixed number of epochs. There is no test-based selection.
Use at least two participants and fresh caches after changing the inputs or
processing. This cross-participant workflow does not reproduce the paper's
within-session, cross-session or adaptation results.

The safe model file stores all constructor arguments and state tensors,
preprocessing and participant/record partition IDs. A JSON sidecar reports
trial counts, optimizer updates and test metrics. Existing outputs are
preserved; strict restoration and inference equality are verified after
freezing constrained convolution weights. The gallery renders source without
running the experiment.
