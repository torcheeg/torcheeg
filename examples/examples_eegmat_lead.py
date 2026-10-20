"""
Probe LEAD through a local author backend on EEGMat
====================================================

This example requires a separately obtained trusted DL4mHealth/LEAD checkout
at c7e21e112a3965b09604b2098e93095880a940dd and the complete official
YiheWang/LEAD P-Base checkpoint, revision8ac42d5a595a3c080c36a6f84932ac1ba10ee857.
Author code and weights use CC-BY-NC-SA4.0; obtain and use them under those
terms. They are not bundled in TorchEEG. The library adapter does not import
or download code; this explicit --backend-root example constructs the local
native model. Cite Wang et al., LEAD, TMLR2026, and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Keep all367official tensors, including sampling-rate vectors50/100/200Hz.
The release has no learned projection head; configure that source module
as Identity, without leaving randomly initialized projection weights.
The adapter returns native contextual features, then a new head learns
background=0 versus mental arithmetic=1. Both disease/subject metadata
columns are zero; no labels or subject identities enter feature extraction.

Resample full records500->200Hz, then MNE FIR0.5--45Hz/notch50Hz. Standardize
each four-second window/channel with population mean/std and floor1e-6.
Keep all19source electrodes in exact order. The native 3D position path
uses the named1020montage; save the actual coordinates in model state.
The ordered-channel fingerprint and physical metadata reject an incompatible
restore, while saved geometry permits an exact restore across a geometry
lookup change. This teaching transfer protocol is not an AD benchmark,
clinical validation, or reproduction of the pretraining schedule.

Hold out the last complete subject and train only a new linear head on all
others for fixed epochs. Save the entire native state, geometry, order hash,
physical metadata, classifier, provenance and held-out predictions. Strictly
restore with the same explicitly supplied backend and reproduce predictions.

Run::

    python examples/examples_eegmat_lead.py --root-path ./eegmat \
        --backend-root ./LEAD --checkpoint ./checkpoint.pth --epochs 1
"""
import argparse
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
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
from torcheeg.models import LEAD, LEADClassifier


VARIANT = 'p-base-external-backend'
SOURCE_REVISION = 'c7e21e112a3965b09604b2098e93095880a940dd'


def build_lead_backend(root, channel_names):
    """Construct a trusted, explicit local author model, without downloading it."""
    root = Path(root).resolve()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned LEAD author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    # The author source has absolute models/layers/utils imports. This example
    # runs in a fresh process with an explicit local checkout supplied by user.
    sys.path.insert(0, str(root))
    from models.LEAD import Model
    available = mne.channels.get_builtin_montages()
    montage = 'colin27_1020' if 'colin27_1020' in available else 'standard_1020'
    config = dict(task_name='pretrain', output_attention=False, patch_len=50, stride=50,
        enc_in=len(channel_names), d_model=128, contrastive_token_ratio=.25,
        channel_names_by_id={0: list(channel_names)}, montage_by_id={0: montage},
        use_sampling_embedding=True, sampling_rate_to_id={50: 0, 100: 1, 200: 2},
        augmentations='none', dropout=.1, n_heads=8, d_ff=256, e_layers=12,
        activation='gelu', max_patch_positions=1024, temporal_pos_type='learnable',
        channel_pos_type='3D', max_channel_positions=368, num_class=2)
    backend = Model(SimpleNamespace(**config))
    backend.projection_head = nn.Identity()  # absent from the actual release
    return backend, config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.5, h_freq=45., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples at200Hz.')
    eeg = np.asarray(eeg, dtype=np.float64)
    return ((eeg-eeg.mean(-1, keepdims=True))/eeg.std(-1, keepdims=True).clip(min=1e-6)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete LEAD P-Base backend EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-lead-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-lead.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_lead_backend(args.backend_root, MENTAL_ARITHMETIC_CHANNEL_LIST)
    backbone = LEAD.from_pretrained(args.checkpoint, backend, MENTAL_ARITHMETIC_CHANNEL_LIST)
    model = LEADClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=800,
        num_channel=19, sampling_rate=200, io_path=args.cache_path, io_mode='pickle',
        num_worker=0, before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(prepare_window), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two complete subject pairs are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
        dataset, subject=subjects[-1]))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.001)
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
    metrics = dict(loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item())
    config = dict(channel_names=list(backbone.channel_names), sampling_rate=backbone.sampling_rate,
        dataset_id=backbone.dataset_id)
    metadata = dict(variant=VARIANT, checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        reference='source EDF reference preserved; no re-referencing applied',
        sampling_rate=200, window_samples=800, input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR0.5--45Hz and50Hznotch on complete recordings AFTER resampling to200Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_backbone=True,
        weights_license='CC-BY-NC-SA-4.0; separate source and checkpoint terms',
        backend_revision=SOURCE_REVISION, backend_config=backend_config, mne_version=str(mne.__version__),
        coordinate_sha256=hashlib.sha256(backbone.electrode_coords.cpu().numpy().tobytes()).hexdigest(),
        native_label_metadata='neutral disease=0/subject=0; physical200Hz/dataset0 only')
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored_config = json.loads(payload['backbone_config_json'])
    restored_backend, _ = build_lead_backend(args.backend_root, restored_config['channel_names'])
    restored = LEADClassifier(LEAD(restored_backend, **restored_config), num_classes=2, freeze_backbone=True).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(payload['state_dict']['backbone.' + name], value, rtol=0, atol=0)
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
