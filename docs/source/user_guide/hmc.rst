HMC recordings and epoch times
==============================

:class:`torcheeg.datasets.HMCDataset` reads the public
`Haaglanden Medisch Centrum sleep staging database
<https://physionet.org/content/hmc-sleep-staging/1.1/>`_. Use the released
recordings directory, containing ``SN001.edf`` and
``SN001_sleepscoring.edf``. The historical dot-separated scoring filename
is also accepted. The reader matches the full SN identifier, checks missing
partners and rejects duplicate/ambiguous EDF files. Text scoring files and
subdirectories are not additional EDF records.

Signals and stages
------------------

The dataset records four EEG, two EOG, chin EMG and ECG channels at 256 Hz.
MNE converts EDF physical values into volts. The reader filters requested
channels before resampling to ``sfreq`` (100 Hz by default). An epoch contains
30 seconds, so the default sample has shape [4, 3000]. Explicit channel order
is preserved. ``missing_channels='zero'`` retains the historical zero-fill
behavior; use ``'raise'`` to require measured channels::

    from torcheeg.datasets import HMCDataset
    from torcheeg import transforms

    dataset = HMCDataset(
        root_path='./HMC/recordings',
        io_path='./cache/hmc-full-v2',
        channels=['EEG C4-M1', 'EOG E1-M2'],
        sfreq=100, l_freq=0.3, h_freq=35.,
        start_at=0., end_at=None, missing_channels='raise',
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Mapping({'Sleep stage W': 0, 'Sleep stage N1': 1,
                                'Sleep stage N2': 2, 'Sleep stage N3': 3,
                                'Sleep stage R': 4})]))

This selects an explicit EEG/EOG pair. It does not derive a bipolar EOG
channel or apply recording normalization. State the reference, channel
selection and preprocessing of the intended experiment. The example is a
reader configuration, not a trained model or reported performance result.

Stages come from the scored W/N1/N2/N3/R event codes. Lights-on/off markers,
movement and unknown intervals do not become sleep-stage labels. The scoring
grid is formed before cropping; a partial scored epoch at either crop edge
is dropped rather than shifted into a new epoch. MNE rejection can remove
additional epochs, whose gaps remain visible in metadata.

Crops, grouping and caches
--------------------------

The historical default still selects ``start_at=900`` through exclusive
``end_at=22500`` (six hours after the initial 15 minutes). These defaults are
not a universal sleep-staging protocol. Use ``start_at=0, end_at=None`` for the
available whole recording. If a requested end exceeds the available signal,
the reader warns and keeps available samples; an empty scored interval raises
an error. A complete last 30-second epoch is retained.

``start_at`` and ``end_at`` metadata are seconds from the original recording
start, including the crop offset. They preserve annotation/rejection gaps.
``subject_id``, ``record_id`` and ``trial_id`` use the released SN identifier;
``clip_id`` is ``SNXXX_<epoch index>`` without source directories. The public
SN number identifies the released night recording. It does not provide a
separate cross-night person linkage. Do not infer one from filenames.

Split recordings before sequence construction. A sequence must stay within
one record and partition and must check adjacent end/start times. The reader
does not pad missing epochs or join a gap. Direct historical ``process_record``
calls without ``epochs_onsets`` still assume explicitly supplied contiguous
epochs beginning at zero; source reads always supply real epoch times.

Use a new ``io_path`` when upgrading this reader. Old caches contain zero-based
crop times, a constant trial ID, and potentially incorrect labels selected
from overlapping marker annotations. Preserve old caches for reproducibility;
they are not silently rewritten. Also use new cache paths after changing the
crop, channels, filters or sampling rate.


Standalone EEGNet training
-------------------------------

:doc:`../auto_examples/examples_hmc_eegnet` trains EEGNet on an explicit
C4-M1 EEG / E1-M2 EOG pair with complete 30-second epochs at 100 Hz. It uses
0.5--30 Hz FIR filtering and per-epoch channel normalization. The default
reads the whole scored record, with optional explicit start/end limits.
Missing measured channels fail rather than being replaced with zeros.

The final sorted public SN group is held out. The same released record
cannot appear in both training and testing, but the public SN identifier
does not prove cross-night person identity. Training uses a fixed epoch
count and no test-based parameter selection. This is a standalone workflow,
not a published benchmark protocol or score.

The example writes a complete EEGNet tensor state plus JSON constructor,
preprocessing and partition metadata. It verifies strict safe loading and
inference agreement. The adjacent JSON report records the actual crop,
record groups, optimizer steps and evaluated epoch counts. Keep an existing
artifact and choose a fresh output path for a new run.

To restore an owned export::

    import json
    import torch
    from torcheeg.models import EEGNet

    payload = torch.load('hmc-eegnet.pt', map_location='cpu', weights_only=True)
    model = EEGNet(**json.loads(payload['model_config_json']))
    model.load_state_dict(payload['state_dict'], strict=True)
    model.requires_grad_(False).eval()

Freezing enables constrained-convolution inference without updating stored
weights. Apply the recorded channel order and preprocessing to new samples;
the model state does not filter, resample or normalize raw EDF signals.
