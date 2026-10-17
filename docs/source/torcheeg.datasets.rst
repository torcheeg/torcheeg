torcheeg.datasets
=========================

.. contents:: The packaged benchmark dataset implementation provides a multi-process preprocessing interface.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.datasets

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

Mother of all BCI Benchmarks
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    moabb.MOABBDataset

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
