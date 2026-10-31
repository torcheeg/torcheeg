"""
Train LGGNet on separate DEAP subjects
=====================================

Use authorized, trusted DEAP ``data_preprocessed_python`` pickle files from
https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ . Deserialization can execute
Python. The provider supplies microvolts, 128 Hz and 4--45 Hz preprocessing.
This example takes the first six stimulus seconds after the three-second
baseline, once per trial. It applies no additional filtering or baseline
subtraction. Preserve the original 32 EEG row order for the built-in region
indices. Fit per-electrode population mean/std over training subjects and time
only. The last sorted subject is held out; valence at least five maps to one.

The full existing TorchEEG LGGNet uses multiscale temporal filters, CBAM,
local-region aggregation, learned global graph and classifier. Published
v1.1.3 initialization, forward behavior and checkpoint keys are retained.
Legacy constructor-only convolution/BatchNorm parameters have no task gradients;
the export records these names rather than claiming every parameter trains.
This is a fixed-epoch teaching protocol, not paper/Arena benchmark reproduction
or a foundation model. Test labels do not select the model. No NNI is required.

Run after installing the checked-out source::

    pip install -e ".[model,dataset]"
    python examples/examples_deap_lggnet.py --root-path ./data_preprocessed_python \
        --epochs 1 --output-path ./deap-lggnet.pt

Export includes full model/optimizer/random states, training-only statistics,
physical/channel/label protocol, input hashes and held-out outputs. Existing
outputs are never overwritten. CLI resume is not implemented.
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
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST, DEAP_GENERAL_REGION_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import LGGNet


CHANNEL_ORDER = list(DEAP_CHANNEL_LIST)


def prepare_features(eeg, baseline):
    if (
        eeg.ndim != 2
        or eeg.shape[0] != 32
        or eeg.shape[1] < 768
        or not np.isfinite(eeg).all()
    ):
        raise ValueError('Expected six finite seconds of 32 EEG rows at 128 Hz.')
    features = eeg[:, :768].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('EEG values must be representable as finite float32.')
    return dict(eeg=features, baseline=baseline)


def standardize_from_train(train_features, test_features):
    mean = train_features.mean((0, 2)).unsqueeze(1)
    scale = train_features.std((0, 2), unbiased=False).unsqueeze(1).clamp_min(1e-6)
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
        description='Train LGGNet on separate DEAP subjects.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./deap-lggnet.pt')
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
    train = TensorDataset(
        train_features.unsqueeze(1), torch.tensor(train_targets, dtype=torch.long)
    )
    test = TensorDataset(
        test_features.unsqueeze(1), torch.tensor(test_targets, dtype=torch.long)
    )
    config = dict(
        region_list=DEAP_GENERAL_REGION_LIST,
        num_electrodes=32,
        chunk_size=768,
        num_classes=2,
    )
    model = LGGNet(**config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        region_indices=DEAP_GENERAL_REGION_LIST,
        feature_dim=model.feature_dim,
        parameters_without_task_gradients=[
            name for name, p in model.named_parameters() if p.grad is None
        ],
        input_preprocessing='provider_preprocessed_4_to_45_Hz_128_Hz',
        baseline_correction=False,
        label_rule='valence_ge_5',
        normalization='per_electrode_train_and_time_population_std_floor_1e-6',
        model_family='LGGNet',
        reference_revision='f3fdf72c0f01f89e0b0760bf5309ebba2f334ccb',
        reference='TorchEEG published v1.1.3 complete LGGNet',
        window_selection='first_six_stimulus_seconds_per_subject_trial',
        source_units='microvolts',
        numpy_version=str(np.__version__),
        optimizer='Adam',
        lr=0.001,
        weight_decay=0.0,
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
