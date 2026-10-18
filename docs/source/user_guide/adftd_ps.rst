ADFTD photostimulation EEG
===========================

Source and diagnosis task
--------------------------

:class:`torcheeg.datasets.ADFTDPSDataset` reads the complementary
`OpenNeuro ds006036 v1.0.5
<https://openneuro.org/datasets/ds006036/versions/1.0.5>`_ photostimulation
release. Its metadata snapshot is Git commit
``dd9fcd3ecc24a91bc27b351ec059fa0e51f59403`` in
`OpenNeuroDatasets/ds006036
<https://github.com/OpenNeuroDatasets/ds006036/tree/dd9fcd3ecc24a91bc27b351ec059fa0e51f59403>`_.
Cite Ntetska et al., Data 10(5), 64 (2025),
https://doi.org/10.3390/data10050064, and the dataset DOI
https://doi.org/10.18112/openneuro.ds006036.v1.0.5.
This is the complementary photostimulation descriptor, distinct from the
2023 resting-state descriptor.

The provider states that this is the same 88-person cohort as ds004504,
with matching participant numbers: 36 AD, 23 FTD and 29 controls.
Keep equal participant IDs together if combining RS and PS records.
PS uses ``task-photomark`` files, not ``task-eyesclosed``. The dataset-level
description identifies eyes-open photostimulation; some inherited README
wording refers to eyes closed. This reader identifies the published task
without inferring an eye state for each window.

The target is diagnosis: ``C -> HC (0)``, ``A -> AD (1)``, ``F -> FTD (2)``
from ``participants.tsv``. ``PHOTO 5Hz`` and ``Photo/HV mark`` acquisition
markers are not diagnosis labels. The reader loads contiguous stored EEG
and does not epoch around these markers, infer per-window stimulation
frequencies or reproduce an SSVEP protocol. A frequency-specific analysis
requires a separately defined event protocol.

Channels, values and preprocessing
-----------------------------------

The canonical output order is
``FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ``.
All source SET headers are continuous 19-channel, 500 Hz records. Header
labels determine the input order; the reader checks/reorders all 19 before
selecting an output subset. The provider's channel table has ``n/a`` type
and unit fields. Output uses the EEGLAB microvolt convention converted to
volts, with ``signal_unit='V'`` and an explicit units-basis field. This format
convention is not independent hardware calibration. Source unit metadata
remain available as JSON.

The default pipeline retains the original added reader's 50 Hz notch,
0.5--45 Hz FIR, output rate 200 Hz and 400-sample windows. To keep native
released samples apart from the conversion to volts, set ``sfreq=500`` and
all three filter settings to None. Additional average reference is explicit
and uses all 19 channels before selecting ``num_channel``.

Use ``preprocessed=False`` for ``sub-XXX/eeg`` original-branch SET files.
Use ``preprocessed=True`` for the provider's cleaned SET files under
``derivatives/eeglab/sub-XXX/eeg``. The BIDS root supplies participants and
matching original channel/JSON sidecars in both modes. The source flag
changes the selected files; configured additional filtering and resampling
still apply. Use the native settings to preserve the provider-cleaned data.
The reader loads the provider's ASR/ICA result and does not recreate it.

Cleaned PS records contain removal boundaries. As with :doc:`adftd_rs`,
filter/resample each contiguous segment separately and never create windows
across a SET ``boundary``. One-based latency ``n + 0.5`` marks the cut after
sample ``n``. A nonpositive ``chunk_size`` returns complete segments, not an
artificial join across removed intervals. Incomplete trailing windows in each
segment are omitted. Short-segment filtering warnings remain visible.

Metadata time is relative to the **stored** SET, including its compressed
clock after source ASR removals. No original-clock reconstruction is claimed.
Output sample bounds are half-open; source interval endpoints are seconds
within that stored recording. Original/cleaned and RS/PS records have distinct
record IDs while retaining the same participant identity. All records from a
person must remain in the same evaluation group.

Load the native original branch
--------------------------------

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import ADFTDPSDataset

    dataset = ADFTDPSDataset(
        root_path='./ADFTD-PS',
        sfreq=500,
        notch_freqs=None,
        l_freq=None,
        h_freq=None,
        chunk_size=500,
        preprocessed=False,
        io_path='./cache/adftd-ps-original-native-1s',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'),
    )
    eeg, label = dataset[0]  # [19, 500], in volts by EEGLAB convention

Missing known participants' task files, mismatched sidecars, unknown diagnosis
codes, malformed/nonfinite SET data or invalid settings fail explicitly.
Matching additional runs/sessions are retained. Hooks and overlapping windows
use independent arrays; before-trial hooks must preserve finite geometry.
Use separate caches when source level or preprocessing changes; see
:doc:`cache`. Graph/grid/region constants are channel-aligned templates,
not measured connectivity. Reader tests and source checks do not establish
clinical classification performance or reproduce a published score.


Regional electrode groups
--------------------------

The general, frontal and hemisphere region lists contain indices into the
unchanged 19-channel source order. For matching the shared regional templates,
``T3/T4/T5/T6`` use the equivalent modern names ``T7/T8/P7/P8``. Each family
includes all 19 source columns once. Signal labels, grid locations and both
adjacency matrices retain their original values. These regional templates
are anatomical priors, not measured patient connectivity.


Train EEGNet
--------------

Use :doc:`/auto_examples/examples_adftd_ps_eegnet` for a standalone three-class
diagnosis workflow. This is continuous photostimulation EEG classification;
``PHOTO``/``Photo/HV`` events are not targets, window limits or inferred
frequency labels. No SSVEP protocol or per-window eye state is claimed.
Keep all 19 channels at 500 Hz, use the EEGLAB microvolt convention to produce
volts, and add no filter, reference or resampling. The channel-table units
are ``n/a``; format conversion is not independent hardware calibration.
Normalize each nonoverlapping one-second window per channel without fitting
statistics across windows or participants.

Hold out the highest numeric participant ID per diagnosis, with all sessions,
runs and windows for that participant in one partition. Require at least two
participants in each class. Train a fixed epoch count without selecting
settings from test labels, and report pooled window cross-entropy, accuracy
and macro F1 for this one split. These workflow metrics are not a published
benchmark or evidence of clinical validity.

.. code-block:: console

    python examples/examples_adftd_ps_eegnet.py --root-path ./ADFTD-PS --epochs 1 --batch-size 32 --output-path ./adftd-ps-original.pt

Pass ``--preprocessed`` to select the provider's ``derivatives/eeglab`` data.
Choose a separate output; the source mode is appended to the cache root.
Windows stay within stored contiguous segments and incomplete segment tails
are omitted. Use a fresh cache root when source files or settings change.
Both source modes share participant identities with each other and with
resting-state ds004504; never treat these alternatives as independent people.

The safe export contains all EEGNet constructor settings, tensor states,
preprocessing and actual participant/record partitions. Load with
``torch.load(..., weights_only=True)``, construct EEGNet from
``model_config_json``, freeze with ``requires_grad_(False).eval()``, then call
``load_state_dict(payload['state_dict'], strict=True)``. A separate JSON report
records window counts, optimizer steps and metrics. Existing outputs are
preserved. The gallery renders this example without executing training.
