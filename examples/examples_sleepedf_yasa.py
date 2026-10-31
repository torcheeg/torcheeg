"""
Score a continuous Sleep-EDF recording with the complete pretrained YASA
========================================================================

Install this feature branch with the optional YASA 0.8.1 dependency under
Python 3.11 or later. Obtain authorized Sleep-EDF Expanded PSG/Hypnogram EDF
pairs from PhysioNet. Pass one complete recording, not separate 30-second
clips. The waveform stays in MNE volts: no extra filtering, standardization
or microvolt conversion is performed. YASA extracts its complete contextual
features, downsamples as needed and uses its original pretrained classifier.
The adapter verifies exact official classifier bytes before loading them.

The fixed example uses EEG Fpz-Cz. YASA recommends a central EEG derivation
when available; this forehead-channel use is a functional transfer example,
not the training montage or a paper/Arena reproduction. The classifier uses
future context in centered temporal features, so inference is offline.

Prediction receives no annotations or stage labels. Attach the Hypnogram only
after prediction, align full 30-second intervals, merge original stages 3/4
as N3 and evaluate only complete scored epochs. W/N1/N2/N3/REM map to
0/1/2/3/4; movement/unscored intervals retain -1. No local model fitting,
threshold selection or held-out-label preprocessing is performed.

Run in an isolated optional environment::

    pip install -e ".[yasa]"
    python examples/examples_sleepedf_yasa.py \
        --psg-path ./SC4001E0-PSG.edf \
        --hypnogram-path ./SC4001EC-Hypnogram.edf \
        --output-path ./sleepedf-yasa.npz

The atomic no-overwrite NPZ contains safe numeric labels/probabilities/onsets,
reference labels/mask and JSON provenance, including exact model/data hashes.
Do not treat a short functional result as a clinical or leaderboard score.
No EEG data, classifiers or arbitrary user-supplied joblib are downloaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import mne
import numpy as np

from torcheeg.models import YASASleepStaging


STAGES = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}


def annotation_labels(annotations, recording_onsets):
    """Match complete physical prediction intervals, without relabeling gaps."""
    labels = np.full(len(recording_onsets), -1, dtype=np.int64)
    for i, onset in enumerate(recording_onsets):
        mask = (annotations.onset <= onset + 1e-6) & (
            annotations.onset + annotations.duration >= onset + 30 - 1e-6
        )
        stages = {
            STAGES[value] for value in annotations.description[mask] if value in STAGES
        }
        if len(stages) > 1:
            raise ValueError('Conflicting full-epoch stage annotations.')
        if stages:
            labels[i] = stages.pop()
    return labels


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_result(payload, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run full-record pretrained YASA on Sleep-EDF.'
    )
    parser.add_argument('--psg-path', required=True)
    parser.add_argument('--hypnogram-path', required=True)
    parser.add_argument('--output-path', default='./sleepedf-yasa.npz')
    args = parser.parse_args()
    psg, scoring, output = (
        Path(args.psg_path),
        Path(args.hypnogram_path),
        Path(args.output_path),
    )
    if output.exists():
        parser.error('output-path exists; select a new result file.')
    if (
        not psg.name.endswith('-PSG.edf')
        or not scoring.name.endswith('-Hypnogram.edf')
        or psg.name[:6] != scoring.name[:6]
    ):
        parser.error(
            'Supply the official paired PSG/Hypnogram filenames for the same subject/night.'
        )
    if not psg.is_file() or not scoring.is_file():
        parser.error('Both authorized EDF files must exist.')
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {path.name: file_hash(path) for path in [psg, scoring]}
    with mne.io.read_raw_edf(
        psg, preload=True, include=['EEG Fpz-Cz'], verbose=False
    ) as raw:
        result = YASASleepStaging(eeg_name='EEG Fpz-Cz').predict(raw)
        # Only evaluation receives ground-truth annotations, after inference.
        raw.set_annotations(mne.read_annotations(scoring), emit_warning=False)
        targets = annotation_labels(raw.annotations, result['recording_onsets'])
    mask = targets >= 0
    if not mask.any():
        raise ValueError('No aligned complete scored epochs found.')
    metadata = dict(
        result['metadata'],
        source_sha256=hashes,
        subject_id=psg.name[:5],
        record_id=psg.name[:-8],
        waveform_preprocessing='none added; continuous MNE volts passed directly to YASA',
        scored_epochs=int(mask.sum()),
        total_prediction_epochs=len(mask),
        labels=STAGES,
        reference_stage_counts=np.bincount(targets[mask], minlength=5).tolist(),
        scored_epoch_accuracy=float(np.mean(result['labels'][mask] == targets[mask])),
        protocol='pretrained offline single-record EEG transfer teaching; no local fitting or paper/Arena score',
        annotation_access='evaluation only, after EEG-only prediction',
        numpy_version=np.__version__,
        mne_version=mne.__version__,
    )
    payload = {
        name: result[name]
        for name in ['labels', 'probabilities', 'onsets', 'recording_onsets']
    }
    payload.update(
        targets=targets, scoring_mask=mask, metadata_json=np.array(json.dumps(metadata))
    )
    save_result(payload, output)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
