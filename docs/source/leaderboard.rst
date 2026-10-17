Leaderboard
===========

Compare results with the same dataset, preprocessing, split and metric.
Each record links to an exact example snapshot and states its evidence type.
Results are grouped by protocol and metric before sorting.

The initial records are **validation runs on two EEGMat subjects with one
training epoch**. They verify runnable workflows. The classification test
subset contains 45 background and 15 arithmetic windows; always predicting
background gives 75% accuracy. Feature-grid reconstruction uses a separate
metric and protocol. Published results will include a primary citation and
their reported evaluation protocol.

.. raw:: html

   <div id="torcheeg-leaderboard" data-source="_static/leaderboard/results.json">
     <p data-status role="status" aria-live="polite">Loading results…</p>
     <div data-controls></div>
     <div data-results></div>
     <noscript>Enable JavaScript to filter the table, or download the JSON below.</noscript>
   </div>

Data and reproducibility
------------------------

Download the :download:`result records <_static/leaderboard/results.json>`.
The records contain source-file hashes, signal units, sample rates, window
sizes, subject IDs, class counts, training settings, source commits, example
checksums and runtime versions. The data come from
`EEG During Mental Arithmetic Tasks <https://physionet.org/content/eegmat/1.0.0/>`_.
Cite Zyma et al. (2019) and PhysioNet. The data use the Open Data Commons
Attribution License v1.0. The immutable source commit identifies the
implementation and its unreleased features. Install that checked-out source with the required optional components
following :doc:`installation`, then use the recorded runtime versions and example
command after downloading the subject pairs.

``validation`` identifies a workflow check, ``measured`` identifies a measured
experiment, and ``published`` identifies a result reported by a cited source.
Use the protocol scope and evidence fields when interpreting a value.
The VAE entry reports the trainer's mean of per-batch reconstruction MSE,
including empty grid cells, on normalized differential-entropy features.

Maintain the records
--------------------

Edit ``docs/source/_static/leaderboard/results.json``. Add a protocol with its
dataset, task, scope, preprocessing and split. Add a uniquely named result
with a finite metric, direction, evidence and an immutable example reference.
Use ``null`` for unreported published parameters rather than inventing them.
Measured and validation runs require their actual data manifest, environment,
command, sample counts, epochs and seed. Published entries require a primary
``citation_url``. Different protocols or metrics form separate comparison
groups.

Check the records and source references from the repository root::

    python docs/validate_leaderboard.py
    python docs/validate_leaderboard.py --verify-sources
    node --test test/utils/test_leaderboard.js

The first command validates the local data without network access. The second
fetches the recorded example files and checks their SHA256 hashes. A successful
documentation build or source-file check does not replace running an experiment.
