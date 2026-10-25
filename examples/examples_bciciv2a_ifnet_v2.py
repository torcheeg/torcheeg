"""
Train IFNetV2 on BCI Competition IV 2a
======================================

Download labelled BNCI2014-001 MAT recordings from
https://bnci-horizon-2020.eu/database/data-sets and retain their attribution.
Use at least two subjects, for example A01T.mat and A02T.mat. Files contain
22 EEG channels at 250 Hz in the provider's channel order, followed by EOG.

Take only the first three-second window two seconds after each trial onset,
skip marked artifacts, and separate the last subject from training. Filter
each window into ordered 4--16 Hz and 16--40 Hz bands with fifth-order causal
Butterworth filters, then divide by five. BandSignal uses the same SciPy
filter design and causal operation as the author preprocessing, applied here
to each extracted window. It does not reproduce the author's longer-epoch
filter history, window selection or full cross-validation/augmentation.

IFNetV2 uses log-power pooling and non-differentiable readout projection.
Train with cross entropy and AdamW at 2**-12 for fixed epochs, without
selection on test labels. This is a CPU teaching protocol, not original-paper
or EEG-Arena performance reproduction. No pretrained checkpoint is used.

Install the checked-out source and run::

    pip install -e ".[model,dataset]"
    python examples/examples_bciciv2a_ifnet_v2.py --root-path ./BCICIV2a \
        --epochs 1 --output-path ./bciciv2a-ifnet_v2.pt

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

import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset
from torcheeg.datasets.constants.motor_imagery.bciciv_2a import BCICIV2A_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import IFNetV2


BAND_SIGNAL = transforms.BandSignal(
    sampling_rate=250, order=5, band_dict={'low': (4, 16), 'high': (16, 40)}
)


def prepare_window(eeg):
    if eeg.shape != (22, 750) or not np.isfinite(eeg).all():
        raise ValueError('Expected a finite 22-channel, three-second window at 250 Hz.')
    bands = BAND_SIGNAL(eeg=eeg)['eeg'].astype(np.float32)
    return bands / 5.0


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
        description='Train IFNetV2 on separate BCI-IV2a subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./bciciv2a-ifnet_v2.pt')
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
        chunk_size=750,
        num_channel=22,
        skip_trial_with_artifacts=True,
        io_mode='memory',
        num_worker=0,
        offline_transform=transforms.Lambda(prepare_window),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose(
            [transforms.Select('label'), transforms.Mapping({1: 0, 2: 1, 3: 2, 4: 3})]
        ),
    )
    # Keep the first cue window, rather than later windows in the same trial.
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
        chunk_size=750,
        num_classes=4,
        num_filters=64,
        num_bands=2,
        temporal_kernel=63,
        patch_size=125,
        dropout=0.5,
        max_norm=0.5,
    )
    model = IFNetV2(**config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2**-12)
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
        chunk_size=750,
        filter_hz=[[4, 16], [16, 40]],
        filter_order=5,
        filter_scope='each_window',
        band_order=['low', 'high'],
        window_selection='first_window_per_subject_session_run_trial',
        normalization='filter_float32_then_divide_by_5',
        model_family='IFNetV2',
        source_units='microvolts',
        numpy_version=str(np.__version__),
        optimizer='AdamW',
        lr=2**-12,
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
