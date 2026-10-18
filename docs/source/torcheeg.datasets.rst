torcheeg.datasets
=========================

.. contents:: The packaged benchmark dataset implementation provides a multi-process preprocessing interface.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.datasets

Disk cache recovery is described in :doc:`user_guide/cache`.

Cached dataset views
--------------------

Data partitioning copies a dataset's metadata and shares its cached signals.
For datasets using lazy IO, each view keeps its own dictionary of record paths
and opens the required record when reading a sample. Copying or splitting a
lazy dataset preserves lazy loading; it does not open all record databases.
Metadata changes in one view do not change another view. Writes to the shared
signal cache are visible to all views, so use separate storage for independent
signal data.

Memory datasets
---------------

With ``io_mode='memory'``, preprocessing keeps each record's signals and
metadata in the dataset object. This mode does not create or reuse a disk
cache, and it does not change existing files at ``io_path``. Each new dataset
instance processes its source records again. Both sequential and parallel
preprocessing return their in-memory records to the dataset. Memory datasets
keep all records available, regardless of ``lazy_threshold``. Dataset copies
and partitions share signal storage while retaining independent metadata.

Emotion Recognition Datasets
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    DEAPDataset
    DREAMERDataset
    SEEDDataset
    SEEDFeatureDataset
    SEEDIVDataset
    SEEDIVFeatureDataset
    SEEDVDataset
    SEEDVFeatureDataset
    AMIGOSDataset
    MAHNOBDataset
    BCI2022Dataset
    MPEDFeatureDataset
    FACEDDataset
    FACEDFeatureDataset

Personal Identification Datasets
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    M3CVDataset

Motor Imagery Datasets
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    BCICIV2aDataset
    PhysioNetMIDataset

PhysioNet motor imagery tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PhysioNetMIDataset`` supports the three label mappings in the
`EEG-Arena task registry <https://github.com/OrangeP0P/eeg-arena/blob/c1aa1292aed21c0e170dd0da3cd0a4a60581f6ad/datasets/__init__.py>`_.
These task mappings do not establish an identical preprocessing or evaluation protocol.

.. list-table::
    :header-rows: 1
    :widths: 20 25 55

    * - ``task``
      - Runs
      - Labels
    * - ``four_class``
      - R04/R06/R08/R10/R12/R14
      - Left fist (0), right fist (1), both fists (2), feet (3).
    * - ``left_right``
      - R04/R08/R12
      - Left fist (0), right fist (1).
    * - ``hands_feet``
      - R04/R06/R08/R10/R12/R14
      - Left, right or both fists (0); feet (1).

Each task has a separate EEGNet training example in the gallery.
All windows from a trial must stay together when partitioning the data.
Use separate cache paths for different tasks. A mismatched task cache raises
an error without replacing its contents. Legacy caches without task metadata
can be read with ``four_class`` only.

Mother of all BCI Benchmarks
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    moabb.MOABBDataset

For source selection, epoch windows and safe raw-data storage, see
:doc:`user_guide/moabb`.

Steady-state Visual Evoked Potential Datasets
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    TSUBenckmarkDataset
    SanDiegoSSVEPDataset

Sleep Stage Detection Datasets 
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    HMCDataset
    ISRUCDataset
    P2018Dataset
    SHHSDataset
    SleepEDFxDataset

Auditory Attention Datasets
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    DTUDataset
    DTUProcessedDataset
    KULDataset
    KULProcessedDataset
    AVEDDataset
    AVEDProcessedDataset

Customized Datasets
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    FolderDataset
    CSVFolderDataset
    MNERawDataset
    ConcatDataset
    PairDataset
    SequenceDataset


Mental Workload Datasets
----------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    MentalArithmeticDataset

Postprocessing hooks
--------------------

``after_trial``, ``after_session`` and ``after_subject`` receive a batch of
windows from the selected group. Its first dimension is the number of windows;
the remaining dimensions are those of one sample. Return the same number and
order of windows so that the signals still match their metadata.

Both NumPy arrays and Torch tensors are split back into individual windows
after the hook. The splitting operation preserves a tensor result's dtype,
device and gradient connections.
For example, this Torch callback centers each electrode's time series within
each window::

    def center_trial(batch):
        return batch - batch.mean(dim=-1, keepdim=True)

Pass it as ``after_trial=center_trial`` when the offline transform returns
tensors, for example with :class:`torcheeg.transforms.ToTensor`. The callback
receives the complete trial batch; the dataset stores its individual windows.
