"""
Train EEGInceptionERP on BNCI2014-008 P300 events
=================================================

Obtain at least two original A01.mat--A08.mat recordings from the BNCI
provider, https://lampx.tugraz.at/~bci/database/008-2014/ . Retain the dataset
attribution: Riccio et al. (2013), doi:10.3389/fnhum.2013.00732. This script
reads local files only and leaves them intact.

The provider stores eight channels in microvolts at 256 Hz. Apply common
average reference and a continuous 1--20 Hz filter. For each rising target
marker, take [0, 1) seconds and resample to 128 samples. Sustained markers
count once. Overlapping flash epochs stay in their original subject.
Use the last subject in filename order only for testing. Fit channel-wise
normalization on training subjects only; use fixed epochs without selection
on test labels. Labels are NonTarget=0 and Target=1.

This short CPU teaching protocol is not the original paper's tuning or
cross-validation experiment, an EEG-Arena score, or a clinical evaluation.
The model is trained from scratch with cross entropy and Adam at 0.001.

Install the source and run::

    pip install -e ".[model,dataset]"
    python examples/examples_bnci2014_008_eeginception_erp.py \
        --root-path ./BNCI2014-008 --epochs 1 --output-path ./erp.pt

The output contains model/optimizer/random states, normalization, held-out
predictions and protocol metadata. Choose a new path each time. The script
does not resume training. No weights or patient data are distributed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
import torch
from torch.utils.data import DataLoader, TensorDataset

from torcheeg.models import EEGInceptionERP


CHANNELS = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']


def load_subject(path):
    run = loadmat(path, struct_as_record=False, squeeze_me=True)['data']
    channels = [str(c) for c in np.asarray(run.channels).reshape(-1)]
    labels = np.asarray(run.y).reshape(-1)
    eeg = np.asarray(run.X, dtype=np.float64)
    classes = [str(c) for c in np.asarray(run.classes).reshape(-1)]
    if (
        channels != CHANNELS
        or eeg.ndim != 2
        or eeg.shape != (len(labels), 8)
        or not np.isfinite(eeg).all()
        or not np.isin(labels, [0, 1, 2]).all()
        or classes != ['NonTarget', 'Target']
    ):
        raise ValueError('Expected original eight-channel BNCI2014-008 P300 data.')
    # A marker spans multiple samples; take only its rising edge.
    onsets = np.flatnonzero((labels > 0) & (np.r_[0, labels[:-1]] == 0))
    onsets = onsets[onsets + 256 <= len(labels)]
    if len(onsets) < 2 or set(labels[onsets]) != {1, 2}:
        raise ValueError('Each subject needs complete target and non-target epochs.')
    eeg = eeg.T
    eeg = eeg - eeg.mean(axis=0, keepdims=True)
    eeg = mne.filter.filter_data(eeg, 256, 1.0, 20.0, verbose=False)
    epochs = np.stack([eeg[:, onset : onset + 256] for onset in onsets])
    epochs = resample_poly(epochs, up=1, down=2, axis=-1).astype(np.float32)
    return epochs, labels[onsets].astype(np.int64) - 1, onsets


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
        description='Train EEGInceptionERP on separate P300 subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./bnci2014-008-eeginception-erp.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size at least two.')
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; select a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.root_path).glob('A0[1-8].mat'))
    if len(files) < 2:
        parser.error('root-path needs at least two distinct A01.mat--A08.mat subjects.')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    subjects = [load_subject(p) for p in files]
    train_x = np.concatenate([s[0] for s in subjects[:-1]])
    train_y = np.concatenate([s[1] for s in subjects[:-1]])
    test_x, test_y, test_onsets = subjects[-1]
    mean = train_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = (
        train_x.std(axis=(0, 2), keepdims=True, dtype=np.float64)
        .clip(min=1e-6)
        .astype(np.float32)
    )
    train_x = torch.from_numpy((train_x - mean) / std)
    test_x = torch.from_numpy((test_x - mean) / std)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(43)
    loader = DataLoader(
        TensorDataset(train_x, torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=order,
    )
    config = {
        'num_electrodes': 8,
        'chunk_size': 128,
        'sampling_rate': 128,
        'num_classes': 2,
    }
    model = EEGInceptionERP(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    updates = 0
    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x) for x in test_x.split(args.batch_size)])
        targets = torch.from_numpy(test_y)
        loss = torch.nn.functional.cross_entropy(logits, targets).item()
        accuracy = (logits.argmax(-1) == targets).float().mean().item()
        balanced_accuracy = np.mean(
            [
                (logits.argmax(-1)[targets == c] == c).float().mean().item()
                for c in [0, 1]
            ]
        )
    metadata = {
        'dataset': 'BNCI2014-008',
        'channels': CHANNELS,
        'source_units': 'microvolts',
        'source_sampling_rate': 256,
        'model_sampling_rate': 128,
        'preprocessing': 'continuous CAR; 1-20 Hz; [0,1)s flash epochs; resample_poly 1/2',
        'normalization': 'training-subject channel mean/std, std floor 1e-6',
        'train_subjects': [p.stem for p in files[:-1]],
        'test_subject': files[-1].stem,
        'source_sha256': hashes,
        'classes': {'NonTarget': 0, 'Target': 1},
        'seed': 42,
        'data_order_seed': 43,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'updates': updates,
        'train_epochs': len(train_y),
        'test_epochs': len(test_y),
        'test_cross_entropy': loss,
        'test_accuracy': accuracy,
        'test_balanced_accuracy': float(balanced_accuracy),
        'protocol': 'fixed-epoch held-out-subject CPU teaching run',
        'torch_version': str(torch.__version__),
        'numpy_version': str(np.__version__),
        'mne_version': str(mne.__version__),
    }
    save_checkpoint(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'data_order_rng_state': order.get_state(),
            'config_json': json.dumps(config, sort_keys=True),
            'metadata_json': json.dumps(metadata, sort_keys=True),
            'normalization_mean': torch.from_numpy(mean),
            'normalization_std': torch.from_numpy(std),
            'test_logits': logits,
            'test_targets': targets,
            'test_onsets': torch.from_numpy(test_onsets.astype(np.int64)),
        },
        output,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
