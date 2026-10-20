BCI Competition IV 2a sample alignment
=======================================

``BCICIV2aDataset`` reads the BNCI 001-2014 MATLAB files (``A01T.mat`` and
matching subject/session files). See the `provider page
<https://bnci-horizon-2020.eu/database/data-sets>`_ and `original description
<https://www.bbci.de/competition/iv/desc_2a.pdf>`_. Inputs contain 22 EEG
and three EOG channels at 250 Hz. Only EEG channels enter classification.

Provider trial onsets are one-based MATLAB sample indices. The reader
converts them to zero-based Python positions before applying ``offset``.
Cached ``start_at`` is zero-based and ``end_at`` is exclusive. The next
trial's onset is an exclusive boundary, so a window cannot consume the
first sample of the next trial. Row and column onset/label/artifact vectors
have identical behavior, including with current NumPy scalar conversion.

For example, provider onset 101 means Python sample 100. ``offset=500``
starts at sample 600, two seconds after trial onset. ``chunk_size=1000``
selects four seconds. The motor imagery cue starts two seconds into each
trial; that explicit window therefore corresponds to the cue interval.
Labels retain original values 1--4 (left hand, right hand, feet, tongue).
Map them to 0--3 explicitly when using four-class cross entropy.

Older caches made before this correction retain the previous alignment.
Use a new cache path to regenerate correctly aligned samples; do not assume
an existing cache was recomputed. Keep all windows from one trial together
when splitting, and use subject isolation for an unseen-subject protocol.
