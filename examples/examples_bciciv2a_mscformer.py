"""
Train MSCFormer on BCI Competition IV 2a
========================================

Download labelled BNCI2014-001 MAT recordings from
https://bnci-horizon-2020.eu/database/data-sets and retain their attribution.
Use at least two subjects, for example A01T.mat and A02T.mat. Files contain
22 EEG channels at 250 Hz in the provider's channel order, followed by EOG.

Filter each recording at 0.5--40 Hz. Start four-second windows two seconds
after each trial onset, skip marked artifacts, and normalize each window per
channel. Keep the last subject in sorted order out of training. Train for a
fixed number of epochs; test labels do not select settings or checkpoints.

The complete Apache-2.0 MSCFormer retains three temporal/spatial CNN
branches, learned positions, the zero class token, five post-norm Transformer
blocks and the full classifier. Use convolution dropout 0.25 and cross entropy
with AdamW at 0.001 for this fixed held-out-subject teaching workflow.

The native reference is the complete Braindecode adaptation at revision
082492fb93a3aa4c5784dc44026642bbc82f49c3. This short CPU run does not reproduce
the original paper's augmentation, five-fold validation or EEG-Arena protocol
and is not a published performance result. No pretrained checkpoint is used.

Install the checked-out source and run::

    pip install -e ".[model,dataset]"
    python examples/examples_bciciv2a_mscformer.py --root-path ./BCICIV2a \
        --epochs 1 --output-path ./bciciv2a-mscformer.pt

The memory dataset leaves the input files intact. The output stores the
complete model, optimizer, random states, test outputs and protocol metadata.
Choose a new output path for each run. This script does not resume training.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset
from torcheeg.datasets.constants.motor_imagery.bciciv_2a import BCICIV2A_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import MSCFormer


def filter_recording(eeg):
    return mne.filter.filter_data(
        eeg, sfreq=250, l_freq=0.5, h_freq=40.0, verbose=False
    )


def prepare_window(eeg):
    if eeg.shape != (22, 1000) or not np.isfinite(eeg).all():
        raise ValueError('Expected a finite 22-channel, four-second window at 250 Hz.')
    return (
        (eeg - eeg.mean(-1, keepdims=True)) / eeg.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def save_checkpoint(payload, path):
    path = Path(path)
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
        description='Train MSCFormer on separate BCI-IV2a subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./bciciv2a-mscformer.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size at least two.')
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; select a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(args.root_path)
    files = sorted(source.glob('*.mat'))
    if not files:
        parser.error('root-path must contain the labelled BCI-IV2a MAT files.')
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    torch.set_num_threads(2)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(43)
    dataset = BCICIV2aDataset(
        root_path=str(source),
        offset=500,
        chunk_size=1000,
        num_channel=22,
        skip_trial_with_artifacts=True,
        io_mode='memory',
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_window), transforms.ToTensor()]
        ),
        label_transform=transforms.Compose(
            [transforms.Select('label'), transforms.Mapping({1: 0, 2: 1, 3: 2, 4: 3})]
        ),
    )
    # Keep only the first cue window per physical trial.
    dataset.info = (
        dataset.info.sort_values(
            ['subject_id', 'session', 'run', 'trial_id', 'start_at'], kind='stable'
        )
        .drop_duplicates(['subject_id', 'session', 'run', 'trial_id'], keep='first')
        .reset_index(drop=True)
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two distinct subjects are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {
            1,
            2,
            3,
            4,
        }:
            raise ValueError(f'Subject {subject} needs all four classes.')
    train, test = next(
        LeaveOneSubjectOut(split_path=str(output) + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    config = dict(
        num_electrodes=22,
        chunk_size=1000,
        num_classes=4,
        num_filters=16,
        temporal_kernels=(85, 65, 45),
        pool_size=44,
        dropout=0.25,
        heads=8,
        depth=5,
        mlp_ratio=4,
        attn_dropout=0.5,
        pos_dropout=0.1,
        final_dropout=0.25,
        attention_scale=None,
    )
    model = MSCFormer(**config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    updates = 0
    for _ in range(args.epochs):
        model.train()
        for eeg, target in DataLoader(
            train, batch_size=args.batch_size, shuffle=True, generator=order
        ):
            optimizer.zero_grad()
            loss = criterion(model(eeg), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in DataLoader(test, batch_size=args.batch_size):
            logits.append(model(eeg))
            targets.append(target)
    logits, targets = torch.cat(logits), torch.cat(targets)
    metrics = dict(
        loss=torch.nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item(),
    )
    if hashes != {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }:
        raise ValueError('Input recordings changed during training.')
    metadata = dict(
        protocol='fixed_epochs_last_subject_holdout',
        epochs=args.epochs,
        batch_size=args.batch_size,
        updates=updates,
        seed=42,
        sampling_rate=250,
        channel_names=BCICIV2A_CHANNEL_LIST,
        offset=500,
        chunk_size=1000,
        filter_hz=[0.5, 40.0],
        normalization='per_window_per_channel_population_std_floor_1e-6',
        model_family='MSCFormer',
        reference_revision='082492fb93a3aa4c5784dc44026642bbc82f49c3',
        reference='braindecode.models.MSCFormer',
        window_selection='first_window_per_subject_session_run_trial',
        source_units='microvolts',
        mne_version=str(mne.__version__),
        numpy_version=str(np.__version__),
        optimizer='AdamW',
        lr=0.001,
        weight_decay=0.01,
        train_subjects=subjects[:-1],
        held_out_subject=subjects[-1],
        train_windows=len(train),
        test_windows=len(test),
        source_sha256=hashes,
        metrics=metrics,
        torch_version=str(torch.__version__),
    )
    save_checkpoint(
        dict(
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            order_rng=order.get_state(),
            config_json=json.dumps(config),
            metadata_json=json.dumps(metadata),
            test_logits=logits,
            test_targets=targets,
        ),
        output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
