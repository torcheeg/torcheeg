"""
Probe complete external Uni-DMFM on EEGMat
==========================================

Explicitly supply the trusted ichenws/Uni-DMFM source at
522f9fc09b6d5f3d1051e6941d1bff548d8f10de and the original separately
obtained pretrained_weights.pth. The pinned source and weight release have
no stated license. No author code or weights are distributed with TorchEEG.
The explicit factory retains the original 242 neural tensors and all 144
saved nonneural profiling buffers; strict loading keeps all 386 states.
All three native reconstruction/contrastive heads remain present.

Use all 19 physical EEGMat sensors in the declared order and original EDF
reference. Resample whole recordings 500->200 Hz, then FIR 0.1--75 Hz and
50 Hz notch; use four-second windows and per-window/channel population
z-scores with a 1e-6 microvolt standard-deviation floor. This is a teaching
transfer protocol. It differs from the author's filter-before-segmentation,
last/first 60-second selection and fixed flatten-token constrained head.
This example explicitly uses mean native tokens and a new linear head.

Hold out a complete subject, train only the new head, export and strictly
restore every native tensor and the physical/provenance configuration.
Do not use held-out labels for parameter selection. Cite Uni-DMFM:
A Universal Dynamic Multi-Scale Foundation Model for EEG Representation
Learning, https://doi.org/10.1016/j.patcog.2026.114221, and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
Install timm separately for the explicit original backend.

Run::

    python examples/examples_eegmat_unidmfm.py --root-path ./eegmat \
        --backend-root ./Uni-DMFM --checkpoint ./pretrained_weights.pth --epochs 1
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
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
from torcheeg.models import UniDMFM, UniDMFMClassifier

VARIANT = 'base-complete-external-backend'
SOURCE_REVISION = '522f9fc09b6d5f3d1051e6941d1bff548d8f10de'
CHECKPOINT_SHA256 = '92fc5f59cd3ffc1c25898f1a573db8d8194935a163bbcb9c68d45f86ba81d5fe'
CHANNELS = [
    'Fp1',
    'Fp2',
    'F3',
    'F4',
    'F7',
    'F8',
    'T3',
    'T4',
    'C3',
    'C4',
    'T5',
    'T6',
    'P3',
    'P4',
    'O1',
    'O2',
    'Fz',
    'Cz',
    'Pz',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_unidmfm_backend(root, checkpoint_path, channel_names):
    """Explicitly construct the full trusted source and released profiling state."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete Uni-DMFM author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError(
            'Use the documented complete original checkpoint for backend construction.'
        )
    name = (
        '_torcheeg_external_unidmfm_'
        + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'models/model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    backend = sys.modules[name].UniDMFM()
    state = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True, mmap=True
    )['model']
    expected = backend.state_dict()
    extra = set(state) - set(expected)
    if len(expected) != 242 or not set(expected) <= set(state) or len(extra) != 144:
        raise ValueError(
            'Require all original neural tensors and the released profiling buffers.'
        )
    for key in expected:
        if (
            state[key].shape != expected[key].shape
            or state[key].dtype != expected[key].dtype
        ):
            raise ValueError('Native neural shape/type mismatch: ' + key)
    for key in sorted(extra):
        path, _, buffer = key.rpartition('.')
        value = state[key]
        if (
            buffer not in ('total_ops', 'total_params')
            or value.shape != (1,)
            or value.dtype != torch.float64
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError('Invalid original nonneural profiling buffer: ' + key)
        target = backend.get_submodule(path) if path else backend
        target.register_buffer(buffer, torch.zeros_like(value))
    return backend, dict(
        num_layers=12,
        nhead=12,
        d_model=384,
        patch_size=200,
        stft_params=dict(backend.stft_params),
        drop_path_rate=0.1,
        profiling_buffers=144,
    )


def filter_recording(eeg):
    values = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=200,
        l_freq=0.1,
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
    values = values[[MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNELS]]
    return (
        (values - values.mean(-1, keepdims=True))
        / values.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Train a frozen complete UniDMFM base backend EEGMat probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-unidmfm-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-unidmfm.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_unidmfm_backend(
        args.backend_root, args.checkpoint, CHANNELS
    )
    backbone = UniDMFM.from_pretrained(args.checkpoint, backend, CHANNELS)
    model = UniDMFMClassifier(backbone, num_classes=2, freeze_backbone=True)
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
        channel_names=list(backbone.channel_names), sampling_rate=backbone.sampling_rate
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
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR 0.1--75 Hz and 50 Hz notch on complete recordings AFTER resampling to 200 Hz',
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
        features='mean of all native 384-wide sensor/time tokens; new task linear head',
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
    restored_backend, _ = build_unidmfm_backend(
        args.backend_root, args.checkpoint, restored_config['channel_names']
    )
    restored = UniDMFMClassifier(
        UniDMFM(restored_backend, **restored_config),
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
