Dataset caches and recovery
===========================

Disk caches store the result of reading recordings, offline transforms, and
postprocessing hooks. Online and label transforms run when a sample is read.
The following behavior applies to dataset caches created by ``BaseDataset``
with ``io_mode='lmdb'``, ``'pickle'``, or ``'memmap'``. Direct use of
:class:`torcheeg.io.EEGSignalIO` does not create a dataset generation.

Create and reuse a cache
------------------------

Choose one ``io_path`` for one source and preprocessing configuration. Create
the dataset again with the same settings to reuse its completed cache. Changing
``online_transform``, ``label_transform``, ``num_worker``, ``verbose``,
``io_size``, or ``lazy_threshold`` does not change offline preprocessing.

New caches record the dataset class, preprocessing parameters, callbacks,
relevant implementation code, and available source-file metadata. TorchEEG
rejects a cache when these recorded settings differ. Use a new ``io_path`` for
a different preprocessing experiment. Keep the previous cache for inspection.

Source checks use file names, sizes, and modification times. They do not hash
the full signal files. An explicit source directory is scanned when the cache
is opened; this can take time for large collections. A completed cache can be
read when its source files are unavailable. It cannot then verify those files.
Changes in external state used by a callback, such as a global variable or an
external service, may not be identifiable. Use a new path when that state or
an untracked source changes.

Resume interrupted preprocessing
---------------------------------

Run the same dataset constructor with the same ``io_path`` and offline settings
after an exception or process exit. TorchEEG retains completed records in an
owned staging directory and processes the unfinished records again. The final
cache directory becomes visible after all records and hooks finish. A process
lock coordinates constructors that use the same local cache path.

Postprocessing hooks work on a separate copy of the completed records. If a
hook fails, the next attempt starts with the clean records. Previously modified
signals do not receive the hook twice. The hook itself can run again: external
side effects, such as writing another file or sending a request, are not rolled
back. Hooks must produce repeatable results for the same inputs. The separate
copy requires additional disk space approximately equal to the processed cache.

Keep the source files available and unchanged while resuming an unfinished
generation. A configuration or source mismatch stops recovery and preserves
the generation. Source files are checked again before publication, but the
cache lock does not prevent another program from modifying them. Do not change
recordings while preprocessing runs. Atomic publication and locking assume a
filesystem that supports the corresponding rename and file-lock operations.

Recognize cache files
---------------------

For ``io_path='/data/cache'``, TorchEEG reserves sibling names
``.cache.torcheeg-lock``, ``.cache.torcheeg-stage``,
``.cache.torcheeg-new-*``, and ``.cache.torcheeg-retired-*``.
Keep these names available for cache coordination and recovery. Do not edit
their manifests or move them into another cache generation.

TorchEEG removes temporary records and hook copies only inside a verified owned
generation. Cleanup of a completed generation can resume after interruption.
An unrecognized staging directory is preserved. Inspect a reported path before
removing anything manually; a different ``io_path`` lets you start an independent
generation without modifying the reported files.

Validation and older caches
---------------------------

New manifests validate metadata checksums, record membership, and the presence
of clip and baseline keys. A missing record, altered metadata, or missing signal
key stops reuse. This validation does not establish the scientific correctness
of preprocessing or detect every possible signal-payload corruption. Signals
are decoded by their selected storage backend when read.

Replacing an existing signal through ``BaseDataset.write_eeg`` remains supported.
Dataset views share that storage. Such replacement does not update the recorded
preprocessing identity; keep a separate cache when an experiment needs the
original signals. Do not edit cached metadata files in place.

Older caches have no generation manifest. TorchEEG checks their readable
structure and referenced signal keys and emits a warning. It does not add a
completion marker or infer that an interrupted legacy cache is complete. If an
older generation was interrupted, preserve it and build a new cache path.

With ``io_mode='memory'``, each constructor processes records in memory. It does
not create or resume a disk generation. See :doc:`../torcheeg.datasets`.
