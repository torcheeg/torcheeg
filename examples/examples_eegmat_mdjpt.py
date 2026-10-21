"""
Probe complete external mdJPT on EEGMat
=======================================

Explicitly supply trusted ncclab-sustech/mdJPT_nips2025 source at
bb84e89df5d05928ef3aa8645cbc339b59ed59de and its complete original
log/pretrain/ckpt/epoch=19.ckpt. The source has no stated license. No author
implementation or pretrained weights are bundled with TorchEEG. Keep all
82 released states, including the complete original MLLA/CNN/projector.

Use all 19 measured EEGMat sensors and explicitly declare the historical
physical aliases T3->T7, T4->T8, T5->P7, T6->P8. Opt in to the author's
fixed neighbor projection for the 41 missing native positions; this is
interpolation, not an assertion that all 60 electrodes were measured.
Resample whole recordings from 500 to 125 Hz, apply FIR 1--47 Hz filtering,
and take the common average over 19 measured
sensors, then five-second population-z-scored windows. No ICA is applied.
These are teaching transfer choices, not full original-paper preprocessing.

The complete source uses half-batch normalization even during inference.
The interface explicitly uses native two-row calls (one window per group),
duplicating/discarding an odd singleton tail. This sample-independent view
is distinct from original multiwindow paired training. The default task head
uses mean 512-wide pre-projector features. All original projection states
remain in the complete export. Hold out the last whole subject, train only
the new head, export physical configuration/hashes/predictions and restore
all states strictly. Gallery rendering does not execute this training.

The explicit factory verifies the source revision, redirects only its
relative interpolation-table load through a module-local NumPy facade, and
installs an independently implemented device-neutral sliding-window helper
in that trusted namespace. Every original neural class/body is retained.
The original helper hardcodes a CUDA allocation. CPU original-loop allocation
port comparisons are separate from actual unmodified GPU validation.
The factory does not change CWD, global NumPy or caller NVML environment.
Use a fresh process if another package named src is already loaded.
Install omegaconf==2.3.0 in this backend environment as well as normal model
and dataset dependencies.

Cite mdJPT, https://arxiv.org/abs/2510.22197, and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Run::

    python examples/examples_eegmat_mdjpt.py --root-path ./eegmat \
        --backend-root ./mdJPT_nips2025 \
        --checkpoint ./mdJPT_nips2025/log/pretrain/ckpt/epoch=19.ckpt --epochs 1
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
from torcheeg.models import MDJPT, MDJPTClassifier
from torcheeg.models.transformer.mdjpt import _patch_windows

VARIANT = 'complete-original-mlla-cnn-external-backend'
SOURCE_REVISION = 'bb84e89df5d05928ef3aa8645cbc339b59ed59de'
CHANNELS = list(MENTAL_ARITHMETIC_CHANNEL_LIST)
ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _RelativeArrayFiles:
    """Forward NumPy symbols; resolve only the pinned table's relative load."""

    def __init__(self, root):
        self.root = root

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, file, *args, **kwargs):
        if str(file) != 'channel_interpolate.npy':
            raise ValueError('Only the declared interpolation table is expected.')
        return np.load(self.root / 'channel_interpolate.npy', *args, **kwargs)


def build_mdjpt_backend(root, checkpoint_path, channel_names):
    """Construct complete trusted source with explicit non-neural device port."""
    from omegaconf import OmegaConf

    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the pinned complete mdJPT original source revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    folder = root / 'src'
    existing = sys.modules.get('src')
    if existing is not None:
        if [Path(v).resolve() for v in getattr(existing, '__path__', [])] != [folder]:
            raise ValueError('Use a fresh process; another src package is loaded.')
    else:
        package = types.ModuleType('src')
        package.__path__ = [str(folder)]
        package.__spec__ = ModuleSpec('src', loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(folder)]
        sys.modules['src'] = package
    previous_nvml = os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK')
    try:
        source = importlib.import_module('src.model.MultiModel_PL')
    finally:
        if previous_nvml is None:
            os.environ.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None)
        else:
            os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = previous_nvml
    source.np = _RelativeArrayFiles(root)
    patch_source = importlib.import_module('src.model.MLLA_new')
    patch_source.to_patch = _patch_windows
    configuration = root / 'log/pretrain/norelu_DATASET_loo_SEEDV.yaml'
    cfg = OmegaConf.load(configuration)
    cfg.data_cfg_list = [
        cfg['data_' + str(i)]
        for i in range(5)
        if cfg['data_' + str(i)].dataset_name != 'None'
    ]
    OmegaConf.resolve(cfg)
    backend = source.MultiModel_PL(cfg=cfg)
    backend._patch_device_policy = 'device_neutral_float32_unfold'
    return backend, dict(
        encoder='complete_original_MLLA_CNN',
        patch_size=32,
        patch_stride=6,
        width=128,
        out_width=32,
        layers=2,
        heads=8,
        native_channels=60,
        projected_channels=64,
        feature_width=512,
        projector_width=1024,
        interpolation_sha256=sha256(root / 'channel_interpolate.npy'),
        configuration_sha256=sha256(configuration),
        patch_device_policy=backend._patch_device_policy,
        physical_aliases={
            k: v
            for k, v in ALIASES.items()
            if k.casefold() in {name.casefold() for name in channel_names}
        },
        missing_native_positions=len(MDJPT.PRETRAINING_CHANNELS) - len(channel_names),
        grouping='one window per native half-group; explicit duplicate/discard odd tail',
    )


def filter_recording(eeg):
    values = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=125,
        l_freq=1.0,
        h_freq=47.0,
        verbose=False,
    )
    return values - values.mean(axis=0, keepdims=True)


def prepare_window(eeg):
    values = np.asarray(eeg, dtype=np.float64)
    if values.shape != (19, 625) or not np.isfinite(values).all():
        raise ValueError(
            'Require 19 finite physical channels and 625 samples at 125 Hz.'
        )
    return (
        (values - values.mean(-1, keepdims=True))
        / values.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Train a frozen complete MDJPT EEGMat probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-mdjpt-125hz-5s')
    parser.add_argument('--output-path', default='./eegmat-mdjpt.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_mdjpt_backend(
        args.backend_root, args.checkpoint, CHANNELS
    )
    backbone = MDJPT.from_pretrained(
        args.checkpoint,
        backend,
        CHANNELS,
        allow_interpolation=True,
        channel_aliases=ALIASES,
    )
    model = MDJPTClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {
        path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))
    }
    dataset = MentalArithmeticDataset(
        root_path=str(source),
        chunk_size=625,
        num_channel=19,
        sampling_rate=125,
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
        allow_interpolation=backbone.allow_interpolation,
        channel_aliases=backbone.channel_aliases,
        feature_mode=backbone.feature_mode,
    )
    metadata = dict(
        variant=VARIANT,
        checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes,
        channel_order=CHANNELS,
        reference='common average over all 19 measured EEG sensors before native interpolation',
        sampling_rate=125,
        window_samples=625,
        input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score, std floor 1e-6 microvolts',
        filters='MNE FIR 1--47 Hz on complete recordings AFTER resampling to 125 Hz; 19-sensor common average reference',
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
        weights_license='Separately obtained original author checkpoint; no license statement found in the pinned source',
        backend_revision=SOURCE_REVISION,
        backend_config=backend_config,
        mne_version=str(mne.__version__),
        features='mean original 512-wide pre-projector feature map; full native projector retained; independent two-row transfer context',
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
    restored_backend, _ = build_mdjpt_backend(
        args.backend_root, args.checkpoint, restored_config['channel_names']
    )
    restored = MDJPTClassifier(
        MDJPT(restored_backend, **restored_config),
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
