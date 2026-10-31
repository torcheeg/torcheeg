"""
Train HSLT on grouped DEAP differential-entropy features
=======================================================

Obtain authorized DEAP ``data_preprocessed_python`` .dat files from
https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ and retain provider attribution.
These Python pickle files must be trusted; deserialization executes Python.
Use at least two subjects. Provider preprocessing retains 32 EEG channels in
microvolts, downsampled to 128 Hz and filtered at 4--45 Hz. No data download
is performed here.

Use the first six stimulus seconds of each trial after the three-second
baseline. Compute four differential-entropy bands with fifth-order causal
Butterworth filters (4--8, 8--14, 14--31, 31--45 Hz), then concatenate rows
in the pinned LibEER nine-region order. The baseline is not subtracted.
Fit each electrode/band mean and population standard deviation on training
subjects only; apply the saved transform to the held-out last sorted subject.
Valence at least five maps to class one; values below five map to zero.

The complete MIT LibEER HSLT has nine electrode Transformers, learned region
projections, a region Transformer and the full classifier. Request logits
explicitly for cross entropy. Four bands are an explicit teaching choice;
this does not reproduce a paper's five-band experiment, or the fixed Arena
raw-signal adaptation. It is a supervised model, not an EEG foundation model.

Run after installing the checked-out source::

    pip install -e ".[model,dataset]"
    python examples/examples_deap_hslt.py --root-path ./data_preprocessed_python \
        --epochs 1 --output-path ./deap-hslt.pt

Training uses fixed epochs; held-out labels do not select a checkpoint.
Export includes the complete model/optimizer/random states, training-only
feature statistics, physical/channel/label protocol, hashes and test outputs.
Existing output is never overwritten. CLI training resume is not implemented.
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

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import HSLT


# Regional order of the pinned LibEER DEAP implementation.
REGIONS = (
    ('FP1', 'AF3', 'AF4', 'FP2'),
    ('F7', 'F3', 'FZ', 'F4', 'F8'),
    ('FC5', 'T7', 'CP5'),
    ('FC1', 'C3', 'CZ', 'C4', 'FC2'),
    ('FC6', 'T8', 'CP6'),
    ('P7', 'P3', 'PO3'),
    ('CP1', 'PZ', 'CP2'),
    ('P8', 'P4', 'PO4'),
    ('O1', 'OZ', 'O2'),
)
CHANNEL_ORDER = [name for region in REGIONS for name in region]
CHANNEL_INDICES = [DEAP_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
BANDS = dict(theta=(4, 8), alpha=(8, 14), beta=(14, 31), gamma=(31, 45))


def prepare_features(eeg, baseline):
    if (
        eeg.ndim != 2
        or eeg.shape[0] != 32
        or eeg.shape[1] < 768
        or not np.isfinite(eeg).all()
    ):
        raise ValueError(
            'Expected at least six finite seconds of 32 EEG rows at 128 Hz.'
        )
    features = transforms.BandDifferentialEntropy(
        sampling_rate=128, order=5, band_dict=BANDS
    )(eeg=eeg[:, :768])['eeg']
    features = features[CHANNEL_INDICES].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Differential entropy requires finite nonzero band variance.')
    return dict(eeg=features, baseline=baseline)


def standardize_from_train(train_features, test_features):
    mean = train_features.mean(0)
    scale = train_features.std(0, unbiased=False).clamp_min(1e-6)
    return (train_features - mean) / scale, (test_features - mean) / scale, mean, scale


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
        description='Train HSLT on separate DEAP subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./deap-hslt.pt')
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
    files = sorted(
        path
        for path in source.glob('*.dat')
        if not path.name.startswith('.') and path.is_file()
    )
    if not files:
        parser.error(
            'root-path must contain trusted DEAP preprocessed Python .dat files.'
        )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    torch.set_num_threads(2)
    torch.manual_seed(42)
    order = torch.Generator().manual_seed(43)
    dataset = DEAPDataset(
        root_path=str(source),
        chunk_size=-1,
        num_channel=32,
        io_mode='memory',
        num_worker=0,
        offline_transform=prepare_features,
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose(
            [transforms.Select('valence'), transforms.Binary(5.0)]
        ),
    )
    ratings = dataset.info.valence.to_numpy(dtype=float)
    if not np.isfinite(ratings).all() or (ratings < 1).any() or (ratings > 9).any():
        raise ValueError('Valence ratings must be finite and between one and nine.')
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two distinct subjects are required.')
    if dataset.info.duplicated(['subject_id', 'trial_id']).any():
        raise ValueError(
            'Only one first-six-second feature window per trial is expected.'
        )
    train, test = next(
        LeaveOneSubjectOut(split_path=str(output) + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    train_features, train_targets = zip(*(train[i] for i in range(len(train))))
    test_features, test_targets = zip(*(test[i] for i in range(len(test))))
    train_features, test_features, feature_mean, feature_scale = standardize_from_train(
        torch.stack(train_features), torch.stack(test_features)
    )
    train = TensorDataset(train_features, torch.tensor(train_targets, dtype=torch.long))
    test = TensorDataset(test_features, torch.tensor(test_targets, dtype=torch.long))
    config = dict(num_electrodes=32, in_channels=4, num_classes=2, return_logits=True)
    model = HSLT(**config)
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
        sampling_rate=128,
        channel_names=CHANNEL_ORDER,
        offset=384,
        chunk_size=768,
        filter_bands_hz=BANDS,
        filter_order=5,
        input_preprocessing='provider_preprocessed_4_to_45_Hz_128_Hz',
        baseline_correction=False,
        label_rule='valence_ge_5',
        normalization='per_electrode_band_train_population_std_floor_1e-6',
        model_family='HSLT',
        reference_revision='dddff9776dbdae21195fe320dff0a5ba61628a18',
        reference='XJTU-EEG/LibEER HSLT independent reimplementation',
        window_selection='first_six_stimulus_seconds_per_subject_trial',
        source_units='microvolts',
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
            feature_mean=feature_mean,
            feature_scale=feature_scale,
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
