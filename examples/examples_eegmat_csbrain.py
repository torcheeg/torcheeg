"""
Probe complete external CSBrain on EEGMat
=========================================

Explicitly supply trusted yuchen2199/CSBrain source at
185aee55b24d0410a830df8dd08d03f675616998 and the separately obtained
complete CSBrain.pth. The pinned source and linked release have no stated
license. No author code or pretrained weights are bundled with TorchEEG.
Retain all 295 native states, including the original reconstruction head.
The adapter adds a pure parameter-free projection view that returns original
pre-projection features and original predictions, preserving all state keys.

Use all 19 physical EEGMat sensors in source order/reference. The interface
maps named electrodes to the original input order and native region/topology
permutation; no electrode is silently renamed, interpolated or padded.
Resample full records 500->200 Hz, then MNE FIR 0.5--75 Hz/50 Hz notch.
Use four-second windows and divide microvolts by 100, as in the author's
pretraining amplitude scaling. Short windows and this transfer preprocessing
are explicit teaching choices, not the original full pretraining benchmark.

Hold out the final complete subject, train only a new mean-token linear head,
export all native states, physical metadata, hashes and predictions, then
strictly restore them. The guide separately explains physically aligned
masking and the original author's different mask/target index convention;
this example performs classification and does not hide that distinction.

Cite CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG
Decoding, https://arxiv.org/abs/2506.23075, and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
The explicit external source also needs threadpoolctl and matplotlib.
Its import-time OMP/BLAS environment edits are restored by this factory.

Run::

    python examples/examples_eegmat_csbrain.py --root-path ./eegmat \
        --backend-root ./CSBrain --checkpoint ./CSBrain.pth --epochs 1
"""

import argparse
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import CSBrain, CSBrainClassifier

VARIANT = 'complete-original-19ch-external-backend'
SOURCE_REVISION = '185aee55b24d0410a830df8dd08d03f675616998'
CHANNELS = list(MENTAL_ARITHMETIC_CHANNEL_LIST)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_csbrain_backend(root, checkpoint_path, channel_names):
    """Explicitly construct the complete trusted original pretraining model."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete CSBrain author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    folder = root / 'models'
    existing = sys.modules.get('models')
    if existing is not None:
        if [Path(v).resolve() for v in getattr(existing, '__path__', [])] != [folder]:
            raise ValueError(
                'Use a fresh process; another models package is already loaded.'
            )
    else:
        package = types.ModuleType('models')
        package.__path__ = [str(folder)]
        package.__spec__ = ModuleSpec('models', loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(folder)]
        sys.modules['models'] = package
    # The original module changes these process settings at import time.
    # Restore the caller's exact environment after loading trusted source.
    keys = ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
    environment = {key: os.environ.get(key) for key in keys}
    try:
        source = importlib.import_module('models.CSBrain')
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    backend = source.CSBrain(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        TemEmbed_kernel_sizes=[(1,), (3,), (5,)],
        brain_regions=list(CSBrain.PRETRAINING_REGIONS),
        sorted_indices=list(CSBrain.PRETRAINING_INDICES),
    )
    return backend, dict(
        layers=12,
        heads=8,
        width=200,
        ffn_width=800,
        patch_samples=200,
        pretraining_patches=30,
        backend_channel_names=list(CSBrain.PRETRAINING_CHANNELS),
        regions=list(CSBrain.PRETRAINING_REGIONS),
        sorted_indices=list(CSBrain.PRETRAINING_INDICES),
        projection_view='input_and_original_output',
    )


def filter_recording(eeg):
    values = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=200,
        l_freq=0.5,
        h_freq=75.0,
        verbose=False,
    )
    return mne.filter.notch_filter(values, Fs=200, freqs=[50.0], verbose=False)


def prepare_window(eeg):
    values = np.asarray(eeg, dtype=np.float64)
    if values.shape != (19, 800) or not np.isfinite(values).all():
        raise ValueError(
            'Require all 19 finite source channels and 800 samples at 200 Hz.'
        )
    return (values / 100.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Train a frozen complete CSBrain EEGMat probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-csbrain-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-csbrain.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_csbrain_backend(
        args.backend_root, args.checkpoint, CHANNELS
    )
    backbone = CSBrain.from_pretrained(args.checkpoint, backend, CHANNELS)
    model = CSBrainClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {
        path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))
    }
    dataset = MentalArithmeticDataset(
        root_path=str(source),
        chunk_size=800,
        num_channel=19,
        sampling_rate=200,
        io_path=args.cache_path,
        io_mode='pickle',
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_window), transforms.ToTensor()]
        ),
        label_transform=transforms.Select('label'),
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two complete subject pairs are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(
        LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    updates = 0
    for _ in range(args.epochs):
        model.train()
        for eeg, target in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(eeg), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in DataLoader(test, batch_size=args.batch_size):
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    metrics = dict(
        loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item(),
    )
    config = dict(
        channel_names=list(backbone.channel_names),
        sampling_rate=backbone.sampling_rate,
        backend_channel_names=list(backbone.backend_channel_names),
        allow_montage_transfer=backbone.allow_montage_transfer,
    )
    metadata = dict(
        variant=VARIANT,
        checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes,
        channel_order=CHANNELS,
        reference='source EDF reference preserved; no re-referencing applied',
        sampling_rate=200,
        window_samples=800,
        input_unit='microvolts before normalization',
        normalization='microvolts divided by 100; no fitted population statistics',
        filters='MNE FIR 0.5--75 Hz and 50 Hz notch on complete recordings AFTER resampling to 200 Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1},
        seed=42,
        held_out_subject=subjects[-1],
        train_subjects=subjects[:-1],
        epochs=args.epochs,
        train_windows=len(train),
        test_windows=len(test),
        updates=updates,
        metrics=metrics,
        optimizer={'name': 'Adam', 'lr': 0.001},
        freeze_backbone=True,
        weights_license='Separately obtained author checkpoint; no license statement found in the pinned source or weight release',
        backend_revision=SOURCE_REVISION,
        backend_config=backend_config,
        mne_version=str(mne.__version__),
        features='mean native pre-projection 200-wide tokens; full reconstruction head retained',
    )
    artifact = dict(
        state_dict=model.state_dict(),
        backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata),
        test_logits=logits,
        test_targets=targets,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored_config = json.loads(payload['backbone_config_json'])
    restored_backend, _ = build_csbrain_backend(
        args.backend_root, args.checkpoint, restored_config['channel_names']
    )
    restored = CSBrainClassifier(
        CSBrain(restored_backend, **restored_config),
        num_classes=2,
        freeze_backbone=True,
    ).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(
            payload['state_dict']['backbone.' + name], value, rtol=0, atol=0
        )
    if source_hashes != {
        path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))
    }:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
