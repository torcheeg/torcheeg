"""
Probe complete external SleepGPT on EEGMat
==========================================

Use an explicitly supplied trusted LordXX505/SleepGPT checkout at
fcbb2c7a6489b5742ab77d66483661e45a4aadf2 and the full Figshare 30626870.v1
checkpoint. Figshare specifies CC-BY 4.0 for weights; the source repository
has no stated license. Obtain those materials separately under their terms;
no author source or weights are distributed with TorchEEG. This example
explicitly constructs the complete original Model_Pre and strictly loads all
257 states, including both original time/frequency reconstruction heads.
The release has no pooler weights; a parameter-free Identity exposes features.
Keep the actual released use_cb=False and drop-path 0.1 configuration.

Use selected physical EEGMat sensors C3,C4,F3,O1,Pz in the original reference.
Explicitly map them to native slots; absent EMG,EOG,Fpz are zero-filled and
masked. No physical bipolar electrode is silently renamed. The native model
uses 100 Hz, 30-second epochs and fifteen 200-sample patches per sensor.
Resample full recordings from 500 to 100 Hz, then FIR 0.5--40 Hz; do not apply a 50 Hz
notch at the 100 Hz Nyquist limit. Standardize each epoch/channel with
population mean/std and floor 1e-6 microvolts. This is a documented teaching
transfer protocol, not the original PSG normalization or paper benchmark.
The model's detached standardized STFT matches the original spectral branch,
using the current complex-valued PyTorch API. Preserve both native class
features and train only a new linear task head.

Hold out the last complete subject with no parameter selection on test labels.
Export and strictly restore every native tensor, physical metadata, trained
head, source/checkpoint hashes and held-out predictions. Cite SleepGPT:
A Unified Time-Frequency Foundation Model for Sleep Decoding, and Zyma et al.
(2019), https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
The explicit author backend also needs lightning, timm, transformers and
nvidia-ml-py in an isolated environment; see the model guide for versions.

Run::

    python examples/examples_eegmat_sleepgpt.py --root-path ./eegmat \
        --backend-root ./SleepGPT --checkpoint ./epoch79.ckpt --epochs 1
"""

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.machinery import ModuleSpec
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
from torcheeg.models import SleepGPT, SleepGPTClassifier


VARIANT = 'large-complete-external-backend'
SOURCE_REVISION = 'fcbb2c7a6489b5742ab77d66483661e45a4aadf2'
CHANNELS = ['C3', 'C4', 'F3', 'O1', 'Pz']


def build_sleepgpt_backend(root, checkpoint_path, channel_names):
    """Explicitly construct the complete original class from a trusted checkout."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete SleepGPT author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    # The source package entrypoints eagerly import unrelated datasets and
    # trainers. Namespace paths keep all actual model/helper source bodies
    # while leaving those entrypoints unexecuted. Never replace another main.
    for name, folder in [
        ('main', root / 'main'),
        ('main.modules', root / 'main/modules'),
    ]:
        existing = sys.modules.get(name)
        if existing is not None:
            if [Path(v).resolve() for v in getattr(existing, '__path__', [])] != [
                folder
            ]:
                raise ValueError(
                    'Use a fresh process; another main package is already loaded.'
                )
        else:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = [str(folder)]
            sys.modules[name] = package
    from main.modules.backbone_pretrain import Model_Pre

    payload = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True, mmap=True
    )
    config = dict(payload['hyper_parameters']['config'])
    expected = dict(
        model_arch='backbone_large_patch200',
        patch_size=200,
        random_choose_channels=8,
        actual_channels=None,
        epoch_duration=30,
        fs=100,
        use_cb=False,
        time_only=False,
        fft_only=False,
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(
            'Require the complete published dual-domain large checkpoint configuration.'
        )
    # Source defaults introduced after release; no neural branch is activated.
    config['loss_names'] = dict(config['loss_names'])
    config['loss_names'].setdefault('Spindle', 0)
    config.setdefault('visual', False)
    config['load_path'] = ''  # disable the source constructor's legacy loader
    backend = Model_Pre(config)
    backend.pooler = nn.Identity()  # no pretrained pooler exists in this release
    backend.first_log_gpu = True  # suppress diagnostic logging only
    metadata = {key: config[key] for key in expected}
    metadata.update(
        mask_ratio=config['mask_ratio'],
        drop_path_rate=config['drop_path_rate'],
        loss_function=config.get('loss_function'),
    )
    return backend, metadata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    return mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=100,
        l_freq=0.5,
        h_freq=40.0,
        verbose=False,
    )


def prepare_window(eeg):
    values = np.asarray(eeg, dtype=np.float64)
    if values.shape != (19, 3000) or not np.isfinite(values).all():
        raise ValueError(
            'Require all 19 finite source channels and 3000 samples at 100 Hz.'
        )
    values = values[[MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNELS]]
    return (
        (values - values.mean(-1, keepdims=True))
        / values.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Train a frozen complete SleepGPT large backend EEGMat probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-sleepgpt-100hz-30s')
    parser.add_argument('--output-path', default='./eegmat-sleepgpt.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_sleepgpt_backend(
        args.backend_root, args.checkpoint, CHANNELS
    )
    backbone = SleepGPT.from_pretrained(args.checkpoint, backend, CHANNELS)
    model = SleepGPTClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {
        path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))
    }
    dataset = MentalArithmeticDataset(
        root_path=str(source),
        chunk_size=3000,
        num_channel=19,
        sampling_rate=100,
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
        sampling_rate=100,
        window_samples=3000,
        input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR 0.5--40 Hz on complete recordings AFTER resampling to100Hz; no notch at Nyquist',
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
        weights_license='CC-BY-4.0 Figshare weights; source repository has no stated license',
        backend_revision=SOURCE_REVISION,
        backend_config=backend_config,
        mne_version=str(mne.__version__),
        features='both native768-wide class tokens concatenated to1536 dimensions',
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
    restored_backend, _ = build_sleepgpt_backend(
        args.backend_root, args.checkpoint, restored_config['channel_names']
    )
    restored = SleepGPTClassifier(
        SleepGPT(restored_backend, **restored_config),
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
