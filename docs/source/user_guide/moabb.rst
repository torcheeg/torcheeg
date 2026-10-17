Use MOABB datasets
==================

:class:`torcheeg.datasets.moabb.MOABBDataset` converts a MOABB dataset and
paradigm into TorchEEG samples. Install the dataset dependency group first.
The adapter uses ``paradigm.get_data`` to obtain each selected subject's
epochs, labels, and actual session/run metadata. It does not assume that all
subjects have the same sessions.

Select preprocessing explicitly
-------------------------------

Set channel selection, event selection, filtering, epoch times, and resampling
on the MOABB paradigm. Confirm its output units with the source dataset.
The adapter does not change those units. A single-filter paradigm returns
epochs shaped ``(epochs, channels, samples)``. Filter-bank output retains its
final filter axis. ``chunk_size=-1`` keeps whole epochs. A positive size creates
sample windows; ``overlap`` must be an integer smaller than that size. Short
trailing segments are omitted.

``before_trial`` receives a copy of each source MNE Raw object before MOABB's
preprocessing and must return an MNE Raw object. The adapter uses a shallow
copy of the source dataset and its raw-reading extension point; it does not
replace methods on the user's dataset. A filter bank may read each raw object
more than once. Make callbacks repeatable. ``offline_transform`` receives each
window after MOABB preprocessing. The normal TorchEEG after-hooks, online
transforms, and label transforms remain available.

Keep epoch identities in partitions
-----------------------------------

Each source epoch has one ``trial_id``. All windows from that epoch share it.
``subject_id``, ``session_id``, ``run_id``, and ``roi_id`` retain the source
subject, session, run, and epoch index within that processed subject.
Session/run IDs are strings. Labels retain MOABB's values; map them explicitly
for the selected loss. Do not treat separate windows from one epoch as
independent train/test trials.

Older adapter caches used run IDs as ``trial_id``. They remain readable with a
legacy-cache warning and a derived in-memory ``run_id``; the adapter does not
rewrite them. Build a fresh cache when the evaluation needs epoch-level trial
isolation. For other cache validation and recovery limits, see :doc:`cache`.

Keep raw downloads separate
---------------------------

With ``download_path=None``, MOABB uses its own configured raw-data location.
An explicit ``download_path`` must be outside ``io_path``. The adapter calls
MOABB's download API for the selected subjects before building a new cache;
MOABB reuses existing source files and registers the requested location in its
dataset configuration. Merely having a download directory does not mean that
every selected subject is present.

The processed cache never owns or deletes that raw directory. With an explicit
path, source-file names, sizes and modification times are included in cache
validation. For a default MOABB-managed location, the generic adapter cannot
discover and validate every source path. Use a new processed cache when those
source files or MOABB's configuration change. A completed cache can be read
without redownloading unavailable raw files.

Run a recording example
-----------------------

:doc:`../auto_examples/examples_bciciv2b_eegnet` trains EEGNet on BCI Competition
IV 2b using the public MOABB ``BNCI2014_004`` reader. It states the channels,
units, sampling rate, event interval, class mapping, and held-out subject.
It runs a fixed training workflow without a search service or benchmark
accuracy claim. Different datasets and paradigms require separate protocol
checks; having a MOABB class name alone does not establish validated support.
