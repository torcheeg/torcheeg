"""
Adapt BiDANN with unlabeled target-subject SEED features
========================================================

Obtain authorized SEED ExtractedFeatures MAT files and label.mat from
https://bcmi.sjtu.edu.cn/home/seed/ and retain provider attribution. Use at
least two subjects, with ordinary subject_date.mat filenames. This workflow
uses provider de_LDS: 62 SEED-ordered EEG rows and five feature bands per
one-second frame. It does not recompute provider filtering or LDS smoothing.

Use only the first nine consecutive feature frames per physical trial.
Fit electrode/band mean and population standard deviation on source subjects
only. Reserve the last sorted subject as the target domain. Its unlabeled
features enter adversarial training; its emotion labels never enter the loss
or checkpoint selection. This is transductive unsupervised domain adaptation,
not evaluation on a subject whose features were never observed.

The full MIT LibEER BiDANN has both hemisphere LSTMs, temporal projections,
complete emotion classifier and three gradient-reversed domain heads. Train
source CE plus all three domain CEs for fixed epochs. Provider emotion
-1/0/1 maps to categorical 0/1/2. The pinned independent reference is
XJTU-EEG/LibEER dddff9776dbdae21195fe320dff0a5ba61628a18; unavailable original-
author code and full paper/Arena scores are not reproduced.

Run after installing the checked-out source::

    pip install -e ".[model,dataset]"
    python examples/examples_seed_bidann.py --root-path ./ExtractedFeatures \
        --epochs 1 --output-path ./seed-bidann.pt

The atomic no-overwrite export includes the complete model/optimizer/random
states, source-fitted feature statistics, source/target protocol, input hashes
and target evaluation outputs. CLI training resume is not implemented.
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

from torcheeg.datasets import SEEDFeatureDataset
from torcheeg.datasets.constants import SEED_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import BiDANN, BiDANNLoss


def first_trial_sequences(dataset):
    """Read one complete nine-frame sequence within each physical trial."""
    sequences, labels, identities = [], [], []
    keys = ['subject_id', 'date', 'session_id', 'trial_id']
    info = dataset.info.sort_values(keys + ['start_at'], kind='stable')
    for identity, group in info.groupby(keys, sort=True):
        first = group.iloc[:9]
        if len(first) != 9 or first.start_at.tolist() != list(range(0, 1800, 200)):
            raise ValueError(
                'Each physical trial needs nine consecutive initial frames.'
            )
        if group.emotion.nunique() != 1:
            raise ValueError('Emotion labels must be constant within each trial.')
        frames = [
            dataset.read_eeg(str(row['_record_id']), str(row['clip_id']))
            for _, row in first.iterrows()
        ]
        value = np.asarray(frames, dtype=np.float32)
        if value.shape != (9, 62, 5) or not np.isfinite(value).all():
            raise ValueError('Expected finite nine-frame, 62-row, five-band features.')
        label = int(first.emotion.iloc[0]) + 1
        if label not in (0, 1, 2):
            raise ValueError('SEED emotion labels must be -1, zero or one.')
        sequences.append(torch.from_numpy(value))
        labels.append(label)
        identities.append([int(x) for x in identity])
    if not sequences:
        raise ValueError('At least one complete physical trial is required.')
    return torch.stack(sequences), torch.tensor(labels), identities


def standardize_from_source(source, target):
    mean = source.mean((0, 1))
    scale = source.std((0, 1), unbiased=False).clamp_min(1e-6)
    return (source - mean) / scale, (target - mean) / scale, mean, scale


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
        description='Adapt complete BiDANN on SEED features.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--output-path', default='./seed-bidann.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size at least two.')
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; select a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_path)
    files = sorted(root.glob('*.mat'))
    if not (root / 'label.mat').is_file():
        parser.error('root-path must contain SEED feature recordings and label.mat.')
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source_order = torch.Generator().manual_seed(43)
    target_order = torch.Generator().manual_seed(44)
    dataset = SEEDFeatureDataset(
        root_path=str(root),
        feature=['de_LDS'],
        num_channel=62,
        io_mode='memory',
        num_worker=0,
    )
    subjects = sorted(int(x) for x in dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two distinct subjects are required.')
    source_set, target_set = next(
        LeaveOneSubjectOut(split_path=str(output) + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    source_features, source_labels, source_ids = first_trial_sequences(source_set)
    target_features, target_labels, target_ids = first_trial_sequences(target_set)
    if len(source_features) < 2 or set(source_labels.tolist()) != {0, 1, 2}:
        raise ValueError(
            'Source subjects need at least two sequences and all three classes.'
        )
    batch_size = min(args.batch_size, len(source_features))
    if len(source_features) % batch_size == 1:
        raise ValueError(
            'Choose a batch-size that avoids a one-sequence source remainder for BN.'
        )
    source_features, target_features, mean, scale = standardize_from_source(
        source_features, target_features
    )
    source_loader = DataLoader(
        TensorDataset(source_features, source_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=source_order,
    )
    # The target loader contains features only; target labels are evaluation-only.
    target_loader = DataLoader(
        TensorDataset(target_features),
        batch_size=batch_size,
        shuffle=True,
        generator=target_order,
    )
    config = dict(
        num_electrodes=62,
        in_channels=5,
        num_classes=3,
        sequence_length=9,
        adversarial_weight=1.0,
        hemisphere_layout='aligned',
    )
    model = BiDANN(**config)
    criterion = BiDANNLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    updates = 0
    for _ in range(args.epochs):
        model.train()
        target_iterator = iter(target_loader)
        for features, labels in source_loader:
            try:
                (target_batch,) = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                (target_batch,) = next(target_iterator)
            optimizer.zero_grad()
            loss = criterion(model(features, target_batch), labels)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    with torch.no_grad():
        logits = torch.cat(
            [model(batch) for batch in target_features.split(batch_size)]
        )
    metrics = dict(
        loss=torch.nn.functional.cross_entropy(logits, target_labels).item(),
        accuracy=(logits.argmax(1) == target_labels).float().mean().item(),
    )
    if hashes != {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }:
        raise ValueError('Input files changed during training.')
    metadata = dict(
        protocol='transductive_unlabeled_last_subject_adaptation_fixed_epochs',
        unlabeled_target_features_used_in_training=True,
        target_emotion_labels_used_in_training=False,
        epochs=args.epochs,
        batch_size=batch_size,
        updates=updates,
        seed=42,
        model_family='BiDANN',
        reference='XJTU-EEG/LibEER independent reimplementation',
        reference_revision='dddff9776dbdae21195fe320dff0a5ba61628a18',
        feature='de_LDS',
        feature_frame_rate_hz=1,
        provider_eeg_sampling_rate_hz=200,
        channel_names=SEED_CHANNEL_LIST,
        feature_bands='five provider bands in source order',
        sequence_length=9,
        sequence_selection='first_nine_frames_per_subject_date_session_trial',
        normalization='source_only_electrode_band_population_std_floor_1e-6',
        label_mapping={'-1': 0, '0': 1, '1': 2},
        criterion='source_CE_plus_left_right_global_domain_CEs',
        domain_weight=1.0,
        optimizer='AdamW',
        lr=0.001,
        weight_decay=0.01,
        source_subjects=subjects[:-1],
        target_subject=subjects[-1],
        source_sequences=len(source_features),
        target_sequences=len(target_features),
        source_trial_ids=source_ids,
        target_trial_ids=target_ids,
        source_sha256=hashes,
        metrics=metrics,
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
    )
    save_checkpoint(
        dict(
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            source_order_rng=source_order.get_state(),
            target_order_rng=target_order.get_state(),
            feature_mean=mean,
            feature_scale=scale,
            config_json=json.dumps(config),
            metadata_json=json.dumps(metadata),
            test_logits=logits,
            test_targets=target_labels,
        ),
        output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
