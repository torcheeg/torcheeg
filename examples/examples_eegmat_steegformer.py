"""
Linear-probe ST-EEGFormer on subject-held-out EEGMat
====================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Yang et al. (ICLR 2026), ST-EEGFormer. Source recordings use
Open Data Commons Attribution License v1.0. Supply two complete subject
pairs. Classify background=0 and arithmetic=1; hold out the last sorted
subject. Epochs are fixed. Do not tune on held-out labels.

The reader returns microvolts and resamples complete recordings to 128 Hz.
Filter complete recordings with an MNE FIR bandpass at .5--45 Hz. Preserve
all 19 source channels and their EEG reference. Standardize each channel
of each four-second window using its own mean and population standard
deviation (ddof=0, floor 1e-6 microvolts). This uses no cross-subject fitted
statistics and removes absolute scale. The native encoder receives 512
samples, split into 32 consecutive 16-sample patches. Explicit channel
indices retain the supplied source order. No hidden 200-to-128 resampling,
unknown-channel replacement, or incomplete-patch truncation occurs.

This is a teaching protocol, not a reproduction of the author's benchmark
preprocessing or EEG-Arena scores. The default SMALL RANDOM encoder has
width32 and two blocks. Optional --checkpoint loads a complete local
canonical small encoder (512-wide, eight blocks). For the author's native
training file, explicitly add --allow-native-checkpoint to allow only
argparse.Namespace metadata with weights_only=True and discard recognized
MAE decoder components. Nothing is downloaded; no pretrained performance
is implied by a one-epoch run.

Freeze every encoder parameter and keep it in evaluation mode. Pool its
PRE-NORM patch tokens, excluding CLS, as in the author's downstream global
pooling. A dropout .1 and linear classifier is trained. This representation
is different from the default encoder's mean of post-norm patch tokens.
Save complete encoder/head state, constructor/preprocessing/channel/partition
settings and every held-out logit/target. Rebuild LinearProbe with the saved
settings and load tensor state strictly. No optimizer state is saved.

Run::

    python examples/examples_eegmat_steegformer.py --root-path ./eegmat --epochs 1

Use a new cache/split path when source files or offline preprocessing change.
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
from torcheeg.models import STEEGFormer
from torcheeg.trainers import ClassifierTrainer


class LinearProbe(nn.Module):
    def __init__(self, encoder, channel_names, num_classes=2, dropout=.1):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.register_buffer('channel_indices', encoder.channel_indices(channel_names))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hid_channels, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        features = self.encoder.forward_features(x, self.channel_indices, normalize=False)
        return self.classifier(self.dropout(features.mean(dim=(1, 2))))


def filter_recording(eeg):
    return mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=128,
        l_freq=.5, h_freq=45., verbose=False)


def normalize_window(eeg):
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True, ddof=0)
    return (eeg - mean) / np.maximum(std, 1e-6)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Linear-probe ST-EEGFormer on subject-held-out EEGMat.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-steegformer-19ch-128hz-4s')
    parser.add_argument('--output-path', default='./eegmat-steegformer.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.allow_native_checkpoint and not args.checkpoint):
        parser.error('epochs/batch-size must be positive; native-checkpoint mode requires --checkpoint.')
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=512, sampling_rate=128, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(normalize_window), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Supply at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    encoder_config = dict(patch_size=16, hid_channels=512, mlp_channels=2048, depth=8, num_heads=8,
        num_channel_embeddings=145, max_patches=511, qkv_bias=True, dropout=0., attention_dropout=0.,
        drop_path_rate=0., norm_eps=1e-6)
    if args.checkpoint:
        encoder = STEEGFormer.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix,
            allow_namespace=args.allow_native_checkpoint, exclude_decoder=args.allow_native_checkpoint, **encoder_config)
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        encoder_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        encoder = STEEGFormer(**encoder_config)
        checkpoint_sha256 = None
    probe_config = dict(channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST, num_classes=2, dropout=.1)
    model = LinearProbe(encoder, **probe_config)
    trainer = ClassifierTrainer(model, num_classes=2, lr=.001,
        accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-steegformer-training'))
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
    torch.save(dict(state_dict=model.state_dict(), encoder_config=encoder_config, probe_config=probe_config,
        sampling_rate=128, window_seconds=4, chunk_size=512, patch_size=16, patches=32,
        normalization=dict(axis='channel-wise time', ddof=0, std_floor_microvolts=1e-6),
        filters=dict(bandpass_hz=(.5, 45.), method='MNE FIR on full recordings after resampling'),
        feature_normalize=False, pooling='mean excluding CLS',
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        allow_native_checkpoint=args.allow_native_checkpoint, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=metrics), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, artifact=str(output)))


if __name__ == '__main__':
    main()
