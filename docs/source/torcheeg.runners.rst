torcheeg.runners
================

Runners execute Data records and yield sample dictionaries incrementally.
MultiprocessRunner uses worker processes; JoblibRunner uses worker threads.
Use ``dataset.extend(runner(data))`` to store their output. ``num_worker=0``
processes records in the calling process.

.. currentmodule:: torcheeg.runners

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseRunner
   SequentialRunner
   MultiprocessRunner
   JoblibRunner
