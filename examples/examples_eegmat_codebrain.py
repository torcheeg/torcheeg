"""
Probe pretrained CodeBrain on EEGMat
=====================================

Load the complete official CodeBrain EEGSSM checkpoint and freeze all encoder
parameters. Train a new flatten/ELU classification head on EEGMat background=0
versus mental arithmetic=1. Cite Ma et al., CodeBrain, ICLR2026, and Zyma et al.
(2019), https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
Source/model license is Apache-2.0; retain the author license and attribution.
Use CodeBrain.pth from YjMajy/CodeBrain revision
bef08d2fdb1759685371cc635aad21ce59163689. There is no automatic download or
random/smaller fallback. The TFDual tokenizer is a separate model and is not
needed for this classifier.

Resample whole records from500 to200Hz, then apply MNE FIR0.3--75Hz and50Hz
notch. Scale microvolts by1/100 and reshape four-second800-sample windows
into [19,4,200] nonoverlapping one-second patches. Preserve the exact EEGMat
source channel order/reference, saved in metadata. Spatial convolution makes
channel order meaningful. This explicit teaching protocol uses EEGMat's50Hz
mains and differs from original pretraining60Hz and record trimming rules;
it does not claim to reproduce the pretraining montage or paper benchmark.

Hold out the last sorted complete subject, train all others for fixed epochs,
and never select settings using held-out labels. Save the complete encoder,
prediction heads, initialized kernel-norm buffers, new classifier, JSON
configuration/provenance, predictions and source/model hashes. Strictly restore
and reproduce predictions. Kernel buffers initialize from weights on first
forward, independently of EEG data; trainable backbone parameters remain frozen.
The new classifier returns two cross-entropy logits. Use a fresh cache if
source recordings or preprocessing change. Keep original records unchanged.

Run::

    python examples/examples_eegmat_codebrain.py --root-path ./eegmat \
        --checkpoint ./CodeBrain.pth --epochs 1
"""
import argparse
import hashlib
import json
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
from torcheeg.models import CodeBrain, CodeBrainClassifier


VARIANT = 'eegssm'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.3, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples at200Hz.')
    return (np.asarray(eeg, dtype=np.float64) / 100.).astype(np.float32).reshape(19, 4, 200)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete pretrained CodeBrain-base EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-codebrain-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-codebrain.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backbone = CodeBrain.from_pretrained(args.checkpoint)
    model = CodeBrainClassifier(backbone, num_electrodes=19, num_patches=4, num_classes=2, freeze_backbone=True)
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
    config = dict(hid_channels=backbone.hid_channels, depth=backbone.depth,
        dropout=backbone.dropout, max_length=backbone.max_length,
        num_time_tokens=backbone.num_time_tokens, num_frequency_tokens=backbone.num_frequency_tokens)
    metadata = dict(variant=VARIANT, checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=200, window_samples=800, input_unit='microvolts before normalization',
        normalization='divide microvolts by100; no fitted population statistics',
        filters='MNE FIR0.3--75Hz and50Hznotch on complete recordings AFTER resampling to200Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_backbone=True,
        weights_license='Apache-2.0; retain author license and attribution')
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = CodeBrainClassifier(CodeBrain(**json.loads(payload['backbone_config_json'])), num_electrodes=19, num_patches=4, num_classes=2, freeze_backbone=True).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
