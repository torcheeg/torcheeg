"""
Probe a frozen CBraMod backbone on EEGMat
=========================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Wang et al. (2025), CBraMod. Source recordings use the Open
Data Commons Attribution License v1.0. Obtain at least two complete subject
pairs. Classify background=0 versus mental arithmetic=1, holding out the
last subject in sorted ID order. Do not tune on held-out labels.

The reader supplies microvolts and resamples whole recordings to 200 Hz.
Filter complete recordings at 0.3--75 Hz and notch 60 Hz using MNE's FIR
filters. Divide microvolts by 100. Explicitly reorder all 19 source channels
into the official CBraMod pretraining order; the source and target orders
are different. Split four-second, 800-sample windows into four 200-sample
patches. These source-derived filters, scale and channel order are explicit.
This teaching task does not reproduce the pretraining corpus's 30-second
segmentation, trimming, artifact rejection or the full EEG-Arena protocol.

Freeze the complete backbone, keep its dropout in evaluation mode, average
its features over electrodes and patches, then train one linear head. The
backbone's reconstruction projection is retained in the state but is not
used by feature extraction. The classifier here is an explicit local probe;
it is not the benchmark's multi-layer task head.

By default use a SMALL RANDOM backbone with width 32 and two blocks. Supply
--checkpoint to safely load a complete local canonical 200-wide, 12-block
checkpoint. The loader validates every key and shape and uses weights-only
deserialization without an unsafe fallback. No model or dataset is downloaded
by this script. The official repository links its author checkpoint at
https://huggingface.co/weighting666/CBraMod. Verify source and settings before
using a local weight file. A one-epoch run checks functionality; it does not
establish pretrained benchmark performance.

Save full tensor state, constructor/preprocessing settings and held-out
logits/targets. Restore the same backbone and probe, then use strict tensor
state loading. Raw EEG reference is retained; channel reordering does not
change the source recording's physical reference.

Run::

    python examples/examples_eegmat_cbramod.py --root-path ./eegmat --epochs 1

For a local canonical checkpoint add --checkpoint ./pretrained_weights.pth.
Keep raw files outside the processed cache and select new cache/split paths
when source recordings or offline filtering change.
"""
import argparse
import hashlib
from pathlib import Path

import mne
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import CBraMod
from torcheeg.trainers import ClassifierTrainer


CHANNEL_ORDER = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.3, h_freq=75, verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[60.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


class FrozenCBraModProbe(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(backbone.hid_channels, 2).to(next(backbone.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (19, 800):
            raise ValueError('Use 19 ordered channels and 800 samples at 200 Hz.')
        features = self.backbone.forward_features(x.reshape(x.shape[0], 19, 4, 200))
        return self.head(features.mean(dim=(1, 2)))


def main():
    parser = argparse.ArgumentParser(description='Train a linear EEGMat probe on a frozen CBraMod backbone.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-cbramod-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-cbramod-probe.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    pl.seed_everything(42, workers=True)
    indices = [MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=800, sampling_rate=200, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.PickElectrode(indices),
            transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Use at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    backbone_config = dict(patch_size=200, out_features=200, hid_channels=200, mlp_channels=800,
        depth=12, num_heads=8, dropout=.1, norm_eps=1e-5, position_kernel=(19, 7))
    if args.checkpoint:
        backbone = CBraMod.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **backbone_config)
        checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    else:
        backbone_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        backbone = CBraMod(**backbone_config)
        checkpoint_sha256 = None
    model = FrozenCBraModProbe(backbone)
    trainer = ClassifierTrainer(model, num_classes=2, lr=.001,
        accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-cbramod-training'))
    test_loader = DataLoader(test, batch_size=args.batch_size)
    metrics = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in test_loader:
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), backbone_config=backbone_config,
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=CHANNEL_ORDER,
        sampling_rate=200, patch_size=200, patches=4, window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.3, 75.), notch_hz=60., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=metrics), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, artifact=str(output)))


if __name__ == '__main__':
    main()
