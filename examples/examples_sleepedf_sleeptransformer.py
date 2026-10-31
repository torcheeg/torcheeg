"""
Train the complete SleepTransformer on subject-held-out Sleep-EDF
================================================================

Obtain authorized Sleep-EDF Expanded EDF pairs from PhysioNet and put paired
files in root-path/sleep-cassette. This fixed teaching workflow reads EEG
Fpz-Cz in volts, filters each complete record at 0.3--40 Hz and resamples to
100 Hz with SleepEDFxDataset. Convert to microvolts, use 30-second epochs,
200-sample symmetric Hamming frames at a 100-sample hop and a 256-point FFT.
Drop DC to retain 128 positive-frequency bins, log the power spectral density.
Each epoch has 29 frames. There is no learned spectral condenser.

Take up to four nonoverlapping 20-epoch contexts per physical recording, at
fixed 60-epoch candidate strides. Reject any context crossing an unscored gap.
No stage-based window selection or padding is used. Hold out the last sorted
subject, including all of that subject's nights. Fit per-frequency/channel
mean and population standard deviation using training subjects only. Map
W/N1/N2/N3/REM to 0/1/2/3/4; source stages 3 and 4 are merged by the reader.

Train all 2,448,517 parameters for fixed epochs with the original batch-summed,
epoch-mean CE and half-squared L2 penalty. Report ordinary mean evaluation CE
and accuracy separately. This is a short functional teaching run, not the
paper's SHHS protocol, pretrained inference or an EEG-Arena benchmark score.

Run after installing this feature branch::

    pip install -e ".[model,dataset]"
    python examples/examples_sleepedf_sleeptransformer.py \
        --root-path ./sleep-edf-database-expanded-1.0.0 \
        --epochs 1 --output-path ./sleeptransformer.pt

The no-overwrite checkpoint contains the complete model/optimizer/RNG, fitted
statistics, physical epoch identities, data hashes and held-out predictions.
CLI training resume is not implemented. No dataset or weights are downloaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.datasets import SleepEDFxDataset
from torcheeg.models import SleepTransformer


STAGES = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4,
}


def spectral_features(eeg):
    """Convert one finite volts epoch [1,3000] into [29,128,1] log PSD."""
    eeg = np.asarray(eeg)
    if eeg.shape != (1, 3000) or not np.isfinite(eeg).all():
        raise ValueError('Expected one finite 100 Hz, 30-second EEG channel in volts.')
    frames = np.stack(
        [eeg[:, start : start + 200] * 1e6 for start in range(0, 2801, 100)]
    )
    window = np.hamming(200)
    spectrum = np.fft.rfft(frames * window, n=256, axis=-1)[..., 1:]
    # One-sided PSD doubles interior bins; Nyquist occurs only once.
    power = np.abs(spectrum) ** 2 / (100 * np.square(window).sum())
    power[..., :-1] *= 2
    return np.log(np.maximum(power, 1e-12)).transpose(0, 2, 1).astype(np.float32)


def select_contexts(dataset, max_contexts=4, context_stride=60):
    """Select fixed candidate positions within recordings, rejecting physical gaps."""
    if max_contexts < 1 or context_stride < 20:
        raise ValueError('Need positive max_contexts and context_stride at least 20.')
    features, labels, identities = [], [], []
    info = dataset.info.sort_values(
        ['subject_id', 'record_id', 'start_at'], kind='stable'
    )
    for (subject, record), group in info.groupby(
        ['subject_id', 'record_id'], sort=True
    ):
        count = 0
        for first in range(0, len(group) - 19, context_stride):
            rows = group.iloc[first : first + 20]
            starts = rows.start_at.to_numpy(dtype=float)
            if not np.allclose(np.diff(starts), 30, rtol=0, atol=1e-6):
                continue
            sequence = []
            for _, row in rows.iterrows():
                eeg = dataset.read_eeg(str(row['_record_id']), str(row['clip_id']))
                sequence.append(spectral_features(eeg))
            targets = [STAGES[str(value)] for value in rows.label]
            features.append(torch.from_numpy(np.stack(sequence)))
            labels.append(torch.tensor(targets))
            identities.append(
                dict(subject=str(subject), record=str(record), onsets=starts.tolist())
            )
            count += 1
            if count == max_contexts:
                break
    if not features:
        raise ValueError('No complete contiguous 20-epoch context found.')
    return torch.stack(features), torch.stack(labels), identities


def standardize_from_train(train, test):
    mean = train.mean((0, 1, 2))
    scale = train.std((0, 1, 2), unbiased=False).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def save_checkpoint(payload, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Train complete SleepTransformer on Sleep-EDF.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./sleeptransformer.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-contexts', type=int, default=4)
    parser.add_argument('--context-stride', type=int, default=60)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.max_contexts) < 1
        or args.context_stride < 20
    ):
        parser.error(
            'Positive epochs/batch-size/max-contexts and context-stride >=20 required.'
        )
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; choose a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_path)
    files = sorted((root / 'sleep-cassette').glob('*.edf'))
    if not files:
        parser.error('root-path/sleep-cassette must contain authorized EDF pairs.')
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    torch.set_num_threads(2)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(43)
    dataset = SleepEDFxDataset(
        root_path=str(root),
        studies=['cassette'],
        channels=['EEG Fpz-Cz'],
        l_freq=0.3,
        h_freq=40,
        sfreq=100,
        missing_channels='raise',
        io_mode='memory',
        num_worker=0,
    )
    features, labels, identities = select_contexts(
        dataset, args.max_contexts, args.context_stride
    )
    subjects = sorted(set(row['subject'] for row in identities))
    if len(subjects) < 2:
        raise ValueError(
            'At least two distinct subjects with complete contexts required.'
        )
    train_indices = [
        i for i, row in enumerate(identities) if row['subject'] != subjects[-1]
    ]
    test_indices = [
        i for i, row in enumerate(identities) if row['subject'] == subjects[-1]
    ]
    train, test, mean, scale = standardize_from_train(
        features[train_indices], features[test_indices]
    )
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    loader = DataLoader(
        TensorDataset(train, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=order,
    )
    config = dict(
        num_electrodes=1,
        num_freqs=128,
        num_frames=29,
        sequence_length=20,
        num_classes=5,
    )
    model = SleepTransformer(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    updates = 0
    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = model.compute_loss(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x) for x in test.split(args.batch_size)])
        evaluation_ce = float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, 5), test_labels.flatten()
            )
        )
        accuracy = float((logits.argmax(-1) == test_labels).float().mean())
    metadata = dict(
        source_revision=model.SOURCE_REVISION,
        train_subjects=subjects[:-1],
        held_out_subject=subjects[-1],
        train_contexts=len(train),
        test_contexts=len(test),
        epochs=args.epochs,
        updates=updates,
        sampling_rate=100,
        epoch_seconds=30,
        channels=['EEG Fpz-Cz'],
        reader_units='V',
        spectrum_units='microvolt squared/Hz',
        filter_hz=[0.3, 40],
        frame_samples=200,
        hop_samples=100,
        fft_samples=256,
        frame_window='symmetric Hamming',
        frequency_bins_hz=(np.fft.rfftfreq(256, 0.01)[1:]).tolist(),
        log_floor=1e-12,
        labels=STAGES,
        normalization='training subjects only, population mean/std per frequency/channel',
        context_stride=args.context_stride,
        max_contexts_per_record=args.max_contexts,
        train_context_ids=[identities[i] for i in train_indices],
        test_context_ids=[identities[i] for i in test_indices],
        source_sha256=hashes,
        test_mean_ce=evaluation_ce,
        test_accuracy=accuracy,
        protocol='short fixed-epoch teaching subject holdout; not paper or Arena score',
        torch_version=str(torch.__version__),
    )
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        loader_rng=order.get_state(),
        mean=mean,
        scale=scale,
        test_features=features[test_indices],
        test_logits=logits,
        test_targets=test_labels,
        config_json=json.dumps(config),
        metadata_json=json.dumps(metadata),
    )
    save_checkpoint(payload, output)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
