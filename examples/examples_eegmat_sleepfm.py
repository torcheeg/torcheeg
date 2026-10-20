"""
Probe complete external SleepFM upstream features on EEGMat
===========================================================

Obtain the author SleepFM-Clinical source/checkpoints separately under its
CC-BY-NC4.0 terms, pinned to2bcbae04c3592f61352addb7ac3d4193f0a3ca25.
Use sleepfm/checkpoints/model_base/best.pt from that checkout; its complete
155-state encoder includes the tokenizer, spatial/context/temporal attention
and all batch-normalization buffers. The interface imports no author code;
this explicit local --backend-root example constructs the trusted model.
Install the author's loguru dependency in the isolated environment as well.

The new task head learns background=0 versus mental arithmetic=1, using
all 19 physical source channels as a single BAS modality. Hold out the last
complete subject, with no parameter selection on test labels. Resample
whole EDF records500->128Hz, then MNE FIR0.5--45Hz/notch50Hz. Standardize
each five-second window/channel with population mean/std and floor1e-6.
This short-context transfer normalization is a teaching protocol, not a
reproduction of original long-context PSG pretraining or clinical scores.
All pretrained weights and buffers remain frozen. Export and strictly
restore the full model, metadata, hashes and held-out predictions.

Cite SleepFM, Nature Medicine, doi:10.1038/s41591-025-04133-4, and Zyma et al.
(2019), https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Run::

    python examples/examples_eegmat_sleepfm.py --root-path ./eegmat \
        --backend-root ./sleepfm-clinical \
        --checkpoint ./sleepfm-clinical/sleepfm/checkpoints/model_base/best.pt --epochs 1"""

import argparse
import hashlib
import json
import subprocess
import sys
import importlib.util
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
from torcheeg.models import SleepFM, SleepFMClassifier


VARIANT = 'upstream-external-backend'
SOURCE_REVISION = '2bcbae04c3592f61352addb7ac3d4193f0a3ca25'


def build_sleepfm_backend(root, channel_names):
    """Explicitly construct the complete separately installed author model."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned SleepFM author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    module_name = (
        'sleepfm_author_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    )
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, root / 'sleepfm/models/models.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    config = json.loads(
        (root / 'sleepfm/checkpoints/model_base/config.json').read_text()
    )
    params = {
        key: config[key]
        for key in [
            'in_channels',
            'patch_size',
            'embed_dim',
            'num_heads',
            'num_layers',
            'pooling_head',
            'dropout',
        ]
    }
    return module.SetTransformer(**params), params


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=128,
        l_freq=0.5,
        h_freq=45.0,
        verbose=False,
    )
    return mne.filter.notch_filter(eeg, Fs=128, freqs=[50.0], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 640) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 640 samples at128Hz.')
    eeg = np.asarray(eeg, dtype=np.float64)
    return (
        (eeg - eeg.mean(-1, keepdims=True)) / eeg.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Train a frozen complete SleepFM upstream backend EEGMat probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-sleepfm-128hz-5s')
    parser.add_argument('--output-path', default='./eegmat-sleepfm.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_sleepfm_backend(
        args.backend_root, MENTAL_ARITHMETIC_CHANNEL_LIST
    )
    backbone = SleepFM.from_pretrained(
        args.checkpoint, backend, MENTAL_ARITHMETIC_CHANNEL_LIST
    )
    model = SleepFMClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {
        path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))
    }
    dataset = MentalArithmeticDataset(
        root_path=str(source),
        chunk_size=640,
        num_channel=19,
        sampling_rate=128,
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
        channel_names=list(backbone.channel_names), modality=backbone.modality
    )
    metadata = dict(
        variant=VARIANT,
        checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes,
        channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        reference='source EDF reference preserved; no re-referencing applied',
        sampling_rate=128,
        window_samples=640,
        input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR0.5--45Hz and50Hznotch on complete recordings AFTER resampling to128Hz',
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
        weights_license='CC-BY-NC-4.0; separate source and checkpoint terms',
        backend_revision=SOURCE_REVISION,
        backend_config=backend_config,
        mne_version=str(mne.__version__),
        features='native128-wide pooled upstream representation for BAS modality',
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
    restored_backend, _ = build_sleepfm_backend(
        args.backend_root, restored_config['channel_names']
    )
    restored = SleepFMClassifier(
        SleepFM(restored_backend, **restored_config),
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
