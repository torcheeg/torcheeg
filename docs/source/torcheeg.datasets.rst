torcheeg.datasets
=========================

.. contents:: The packaged benchmark dataset implementation provides a multi-process preprocessing interface.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.datasets

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
    AMIGOSDataset
    MAHNOBDataset
    BCI2022Dataset
    MPEDFeatureDataset

Personal Identification Datasets
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    M3CVDataset

Steady-state Visual Evoked Potential Datasets
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    TSUBenckmarkDataset

Customized Datasets
---------------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: datasettemplate.rst

    NumpyDataset
    MNEDataset
    FolderDataset

Hooks
---------------------------------------------------------

.. autofunction:: before_trial_normalize
.. autofunction:: after_trial_normalize
.. autofunction:: after_trial_moving_avg