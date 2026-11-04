torcheeg.io
===================

.. contents:: A set of unified input and output API is used to store the processing results of various EEG databases for more efficient and convenient use.
    :depth: 2
    :local:
    :backlinks: top

EEG backends
------------

.. currentmodule:: torcheeg.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseEEGIO
   MemoryEEGIO
   LMDBEEGIO
   PickleEEGIO
   NumpyEEGIO
   ZarrEEGIO

Metadata backends
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseInfoIO
   MemoryInfoIO
   CSVInfoIO
   JSONLInfoIO
   SQLiteInfoIO
