torcheeg.transforms
===========================

.. contents:: Extensive EEG preprocessing methods help users extract features, construct EEG signal representations, and connect to commonly used deep learning libraries.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.transforms

Datatype-independent Transforms
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    Compose
    Lambda
    BaselineRemoval

.. currentmodule:: torcheeg.transforms

Numpy-based Transforms
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    CWTSpectrum
    BandSignal
    BandDifferentialEntropy
    BandPowerSpectralDensity
    BandMeanAbsoluteDeviation
    BandKurtosis
    BandSkewness
    DWTDecomposition
    BandApproximateEntropy
    BandSampleEntropy
    BandSVDEntropy
    BandDetrendedFluctuationAnalysis
    BandPetrosianFractalDimension
    BandHiguchiFractalDimension
    BandHjorth
    BandHurst
    BandBinPower
    ARRCoefficient
    BandSpectralEntropy
    PearsonCorrelation
    PhaseLockingCorrelation
    MeanStdNormalize
    MinMaxNormalize
    PickElectrode
    To2d
    ToGrid
    ToInterpolatedGrid
    Concatenate
    MapChunk
    Downsample
    RearrangeElectrode
    Flatten

.. currentmodule:: torcheeg.transforms

PyG-based Transforms
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    pyg.ToG
    pyg.ToDynamicG

.. currentmodule:: torcheeg.transforms

Torch-based Transforms
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    ToTensor
    Resize
    RandomNoise
    RandomMask
    RandomWindowSlice
    RandomWindowWarp
    RandomPCANoise
    RandomFlip
    RandomSignFlip
    RandomShift
    RandomChannelShuffle
    RandomFrequencyShift

.. currentmodule:: torcheeg.transforms

Label Transforms
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    Select
    FixCategory
    Binary
    BinaryOneVSRest
    BinariesToCategory
    StringToInt
    Mapping

.. contents:: We also provide hooks for preprocessing signals, which can leverage global information from trials, sessions, and subjects, and thus can improve the performance of prediction models, especially generalization. Please refer to the paper you are comparing to determine whether you should use it to conduct a fair comparison.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.transforms
    
Hooks
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: transformtemplate.rst

    before_hook_normalize :noindex:
    after_hook_normalize :noindex:
    after_hook_running_norm :noindex:
    after_hook_linear_dynamical_system :noindex: