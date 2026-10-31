"""
Train the complete SlimSeiz network on binary BCI-IV2a
======================================================

Download labelled BNCI2014-001 MAT recordings from
https://bnci-horizon-2020.eu/database/data-sets and retain attribution.
Use at least two subjects. Retain left/right-hand classes (provider 1/2 mapped
to 0/1) and all 22 EEG rows in provider order, 250 Hz, microvolts. The original
binary classifier is unchanged. This explicit transfer workflow does not
perform the seizure framework's adaptive channel selection.

Filter recordings at 0.5--40 Hz, take only the first four-second cue window
per trial (two to six seconds after onset), skip marked artifacts, standardize
each window per channel and hold out the last sorted subject. Use fixed
epochs of cross entropy/AdamW without checkpoint selection on test labels.

Supply the author's trusted original Loop_ADDNet_kFold.ipynb from
https://github.com/guoruilu/SlimSeiz at revision
84b168278173a0e14163a6839fb36d7c41bc3c52 separately. Its repository does not
state a license; no author source/weights are bundled or downloaded. Verify
the exact notebook hash and execute only the two complete neural-definition
cells: multiscale CNN, full pure-PyTorch selective Mamba/RMSNorm residual,
temporal mean and original two-class head. No data/training cells execute.

This is a short motor-imagery transfer teaching run, not seizure prediction,
preictal/interictal validation, supervised-contrastive training, paper scores
or fixed EEG-Arena protocol reproduction. No pretrained checkpoint is used.

Install the checked-out source and run::

    pip install -e ".[model,dataset]"
    python examples/examples_bciciv2a_slimseiz.py --root-path ./BCICIV2a \
        --author-source ./author/Loop_ADDNet_kFold.ipynb --epochs 1 \
        --output-path ./bciciv2a-slimseiz.pt

The atomic no-overwrite export includes complete model/optimizer/random states,
test outputs, input hashes and physical/channel/label/split/source metadata.
CLI training resume is not implemented.
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
from torcheeg.models import SlimSeiz


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
        description='Train SlimSeiz on separate BCI-IV2a subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument(
        '--author-source',
        required=True,
        help='Trusted original Loop_ADDNet_kFold.ipynb notebook.',
    )
    parser.add_argument('--output-path', default='./bciciv2a-slimseiz.pt')
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
            [transforms.Select('label'), transforms.Mapping({1: 0, 2: 1})]
        ),
    )
    # Retain only the first cue window for every subject/session/run/trial.
    dataset.info = (
        dataset.info.sort_values(
            ['subject_id', 'session', 'run', 'trial_id', 'start_at'], kind='stable'
        )
        .drop_duplicates(['subject_id', 'session', 'run', 'trial_id'], keep='first')
        .reset_index(drop=True)
    )
    # Retain the physical left/right-hand classes for the unchanged binary head.
    dataset.info = dataset.info.loc[dataset.info.label.isin([1, 2])].reset_index(
        drop=True
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two distinct subjects are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {
            1,
            2,
        }:
            raise ValueError(f'Subject {subject} needs both left/right-hand classes.')
    train, test = next(
        LeaveOneSubjectOut(split_path=str(output) + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    config = dict(num_electrodes=22, chunk_size=1000, num_classes=2)
    model = SlimSeiz.from_source(args.author_source, **config)
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
        model_family='SlimSeiz',
        retained_labels=[1, 2],
        label_mapping={'1': 0, '2': 1},
        channel_selection='all_22_provider_rows_no_adaptive_selection',
        training_objective='cross_entropy_only_not_supervised_contrastive',
        author_revision=SlimSeiz.SOURCE_REVISION,
        author_source_sha256=SlimSeiz.SOURCE_SHA256,
        output_kind='logits',
        criterion='CrossEntropyLoss',
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
