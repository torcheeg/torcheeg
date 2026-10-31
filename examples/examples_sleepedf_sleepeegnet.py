"""
Train complete bidirectional SleepEEGNet and decode without target labels
=========================================================================

Obtain authorized Sleep-EDF Expanded EDF pairs from PhysioNet and place pairs
in root-path/sleep-cassette. SleepEDFxDataset reads EEG Fpz-Cz in volts,
filters each full record at 0.3--40 Hz and resamples to 100 Hz. Use complete
30-second epochs, convert to microvolts and fit a single channel's mean and
population standard deviation on training subjects only.

Choose up to four nonoverlapping ten-epoch contexts evenly across the valid
60-epoch-stride candidates in each recording. Reject unscored gaps. This time-
based rule does not select using stage labels. Hold out the last sorted
subject with all nights. W/N1/N2/N3/REM map to 0/1/2/3/4; the reader merges
original stages 3 and 4. SOD=5 and EOD=6 are decoder tokens, not sleep stages.

Train the full bidirectional 5,107,206-parameter CNN/encoder/attention-decoder
for fixed epochs, using per-sequence class-mean CE plus non-bias L2. Teacher
forcing uses training labels only. Greedy test inference takes EEG only.
Count missing/invalid stage outputs as wrong in per-input-epoch accuracy and
report stage coverage. This short SGD teaching run does not reproduce the
paper's complete dataset preparation, RMSProp optimizer or performance.
The original TensorFlow source has academic/non-commercial terms and is not
bundled; the library contains a newly written full PyTorch architecture.

Run after installing this feature branch::

    pip install -e ".[model,dataset]"
    python examples/examples_sleepedf_sleepeegnet.py \
        --root-path ./sleep-edf-database-expanded-1.0.0 \
        --epochs 1 --output-path ./sleepeegnet.pt

Export complete model/optimizer/RNG, source-only statistics, source hashes,
physical identities and autoregressive predictions atomically without
overwriting a checkpoint. No CLI training resume, beam decoding, automatic
data/weight download or paper/Arena leaderboard score is claimed.
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
from torcheeg.models import SleepEEGNet


STAGES = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4,
}


def select_contexts(dataset, max_contexts=4, context_stride=60):
    """Keep physical ten-epoch contexts, sampling across complete recordings."""
    if max_contexts < 1 or context_stride < 10:
        raise ValueError(
            'Positive max_contexts and context_stride at least ten required.'
        )
    signals, targets, identities = [], [], []
    info = dataset.info.sort_values(
        ['subject_id', 'record_id', 'start_at'], kind='stable'
    )
    for (subject, record), group in info.groupby(
        ['subject_id', 'record_id'], sort=True
    ):
        candidates = []
        for start in range(0, len(group) - 9, context_stride):
            rows = group.iloc[start : start + 10]
            if np.allclose(
                np.diff(rows.start_at.to_numpy(float)), 30, rtol=0, atol=1e-6
            ):
                candidates.append(rows)
        for index in np.linspace(
            0, len(candidates) - 1, min(max_contexts, len(candidates)), dtype=int
        ):
            rows = candidates[index]
            epochs = []
            for _, row in rows.iterrows():
                value = np.asarray(
                    dataset.read_eeg(str(row['_record_id']), str(row['clip_id']))
                )
                if value.shape != (1, 3000) or not np.isfinite(value).all():
                    raise ValueError('Expected finite [1,3000] volts at 100 Hz.')
                epochs.append((value * 1e6).astype(np.float32))
            signals.append(torch.from_numpy(np.stack(epochs)))
            targets.append(torch.tensor([STAGES[str(stage)] for stage in rows.label]))
            identities.append(
                dict(
                    subject=str(subject),
                    record=str(record),
                    onsets=rows.start_at.astype(float).tolist(),
                )
            )
    if not signals:
        raise ValueError('At least one complete ten-epoch context is required.')
    return torch.stack(signals), torch.stack(targets), identities


def standardize_from_train(train, test):
    mean = train.mean((0, 1, 3), keepdim=True)
    scale = train.std((0, 1, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def align_predictions(tokens, lengths, epochs, num_classes=5):
    """Align generated positions; EOD and missing/invalid tokens remain -1."""
    result = tokens.new_full((tokens.shape[0], epochs), -1)
    count = min(tokens.shape[1], epochs)
    candidates = tokens[:, :count]
    positions = torch.arange(count, device=tokens.device)[None, :]
    valid = (
        (positions < lengths[:, None]) & (candidates >= 0) & (candidates < num_classes)
    )
    # EOD is outside stage IDs and is never treated as a valid sleep prediction.
    result[:, :count] = torch.where(valid, candidates, torch.full_like(candidates, -1))
    return result


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
        description='Train complete SleepEEGNet and infer EEG-only.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./sleepeegnet.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-contexts', type=int, default=4)
    parser.add_argument('--context-stride', type=int, default=60)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.max_contexts) < 1
        or args.context_stride < 10
    ):
        parser.error(
            'Positive epochs/batch-size/max-contexts and context-stride >=10 required.'
        )
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; select a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((Path(args.root_path) / 'sleep-cassette').glob('*.edf'))
    if not files:
        parser.error('root-path/sleep-cassette must contain authorized EDF pairs.')
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    torch.set_num_threads(2)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(43)
    dataset = SleepEDFxDataset(
        root_path=args.root_path,
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
        chunk_size=3000, sequence_length=10, num_classes=5, bidirectional=True
    )
    model = SleepEEGNet(**config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    updates = 0
    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            decoder, target = model.prepare_targets(y)
            optimizer.zero_grad()
            loss = model.compute_loss(model(x, decoder), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    # All held-out EEG contexts are decoded together, so post-EOD raw outputs
    # match the reference's batch-dependent stopping rule. Metrics use lengths.
    generated = model.generate(test)
    predictions = align_predictions(generated['tokens'], generated['lengths'], 10)
    accuracy = float((predictions == test_labels).float().mean())
    coverage = float((predictions >= 0).float().mean())
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
        model_units='microvolt before standardization',
        filter_hz=[0.3, 40],
        normalization='channel-wise mean and population std from training subjects only',
        labels=STAGES,
        start_token=5,
        end_token=6,
        bidirectional=True,
        context_selection='evenly spaced valid fixed-stride candidates across each record; label independent',
        context_stride=args.context_stride,
        max_contexts_per_record=args.max_contexts,
        train_stage_counts=torch.bincount(train_labels.flatten(), minlength=5).tolist(),
        test_stage_counts=torch.bincount(test_labels.flatten(), minlength=5).tolist(),
        train_context_ids=[identities[i] for i in train_indices],
        test_context_ids=[identities[i] for i in test_indices],
        source_sha256=hashes,
        test_accuracy_including_missing=accuracy,
        test_stage_coverage=coverage,
        protocol='short fixed-epoch EEG-only autoregressive subject holdout; not paper or Arena score',
        optimizer='SGD lr=.01; not paper RMSProp',
        torch_version=str(torch.__version__),
    )
    save_checkpoint(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            loader_rng=order.get_state(),
            mean=mean,
            scale=scale,
            test_features=features[test_indices],
            test_tokens=generated['tokens'],
            test_lengths=generated['lengths'],
            test_logits=generated['logits'],
            test_predictions=predictions,
            test_targets=test_labels,
            config_json=json.dumps(config),
            metadata_json=json.dumps(metadata),
        ),
        output,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
