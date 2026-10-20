"""
Probe pretrained EEGMamba on EEGMat
===================================

Load all117tensors of the official EEGMamba checkpoint, freeze the complete
backbone and train its new flatten/ELU classifier on EEGMat background=0
versus mental arithmetic=1. Cite Wang et al., An EEG foundation model with
Mamba (Neural Networks2025), and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
EEGMamba source and official weights use MIT; portable Mamba2/SSD adaptation
retains the upstream Apache-2.0 license and attribution. Download
pretrained_EEGMamba.pth from weighting666/EEGMamba revision
0b060d87acd6f23bf1d0b852bf1726064f335f97. No automatic download or random fallback.

Resample whole records500->200Hz, then MNE FIR.3--75Hz/notch50Hz; scale
microvolts/100 and reshape four-second windows to[19,4,200]. Keep source
channel order/reference explicit. Spatial convolution and alternating
sequence reversals make ordering meaningful. This teaching protocol uses
EEGMat50Hz mains and differs from pretraining60Hz/trimming/montage rules;
it does not reproduce paper benchmarks. Hold out the last complete subject
and use fixed training settings without selecting on heldout labels.

The differentiable PyTorch SSD path needs no CUDA extension. Save the entire
backbone including its reconstruction head, classifier, configuration,
source/checkpoint hashes and heldout predictions; strict restore must reproduce
the predictions and all frozen state. Keep original records unchanged.

Run::

    python examples/examples_eegmat_eegmamba.py --root-path ./eegmat \
        --checkpoint ./pretrained_EEGMamba.pth --epochs 1
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
from torcheeg.models import EEGMamba, EEGMambaClassifier


VARIANT = 'default'


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
    parser = argparse.ArgumentParser(description='Train a frozen complete pretrained EEGMamba-base EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-eegmamba-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-eegmamba.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backbone = EEGMamba.from_pretrained(args.checkpoint)
    model = EEGMambaClassifier(backbone, num_electrodes=19, num_patches=4, num_classes=2, freeze_backbone=True)
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
    config = dict(num_layers=backbone.num_layers, state_size=backbone.state_size,
        head_dim=backbone.head_dim, block_size=backbone.block_size)
    metadata = dict(variant=VARIANT, checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=200, window_samples=800, input_unit='microvolts before normalization',
        normalization='divide microvolts by100; no fitted population statistics',
        filters='MNE FIR0.3--75Hz and50Hznotch on complete recordings AFTER resampling to200Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_backbone=True,
        weights_license='MIT; Mamba2/SSD adaptation Apache-2.0; retain author licenses and attribution')
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = EEGMambaClassifier(EEGMamba(**json.loads(payload['backbone_config_json'])), num_electrodes=19, num_patches=4, num_classes=2, freeze_backbone=True).eval()
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
