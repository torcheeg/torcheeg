"""
Linear-probe LaBraMEncoder on subject-held-out EEGMat
=====================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Jiang et al. (ICLR 2024), LaBraMEncoder. Source recordings use
Open Data Commons Attribution License v1.0. Supply at least two complete
subject pairs. Classify background=0 and arithmetic=1; hold out the last
sorted subject. Fix epochs; do not tune on held-out labels.

The EDF reader returns microvolts and resamples complete recordings to
200 Hz. Apply .1--75 Hz bandpass and 50 Hz notch MNE FIR filters to complete
recordings, preserve all 19 source channels/reference/order, then divide
microvolts by 100. Four-second windows contain four native 200-sample patches.
No per-window z-score or fitted population statistics are applied.

Freeze every encoder parameter and keep it in evaluation mode. Pool raw
patch tokens over electrodes and time, excluding CLS, then apply the
learned native final norm and a fresh dropout .1/linear classifier. Native
pretraining norm weights are reused explicitly; the author's fine-tuning
fc_norm starts separately. This teaching protocol does not reproduce the
published fine-tuning benchmark or an EEG-Arena score.

The default RANDOM encoder has width100/conv4/two blocks. Optional local
--checkpoint uses the canonical base width200/conv8/12-block encoder.
For the author's training checkpoint, additionally set
--checkpoint-prefix student. --allow-native-checkpoint to opt into known
training metadata and removal of recognized student pretraining heads,
always with weights_only=True. Nothing is downloaded. A one-epoch run
does not establish pretrained or benchmark performance.

Save complete state, constructor/preprocessing/channel/partition settings
and every held-out logit/target. Rebuild LinearProbe with both saved
configurations and load state strictly. No optimizer state is saved.
Select new cache/split paths when sources or offline preprocessing change.

Run::

    python examples/examples_eegmat_labram.py --root-path ./eegmat --epochs 1
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
from torcheeg.models import LaBraMEncoder
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
        features = self.encoder(x, self.channel_indices)
        return self.classifier(self.dropout(features))


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.1, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Linear-probe ST-EEGFormer on subject-held-out EEGMat.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-labram-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-labram.pt')
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
        io_mode='pickle', num_channel=19, chunk_size=800, sampling_rate=200, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Supply at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    encoder_config = dict(patch_size=200, hid_channels=200, conv_channels=8, mlp_channels=800,
        depth=12, num_heads=10, qkv_bias=False, qk_norm=True, dropout=0., attention_dropout=0.,
        drop_path_rate=0., layer_scale=.1, norm_eps=1e-6, use_abs_pos_emb=True,
        max_patches=16, interpolate_time=False, pooling='mean')
    if args.checkpoint:
        encoder = LaBraMEncoder.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix,
            allow_training_metadata=args.allow_native_checkpoint, exclude_pretraining=args.allow_native_checkpoint, **encoder_config)
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        encoder_config.update(hid_channels=100, conv_channels=4, mlp_channels=200, depth=2, num_heads=4)
        encoder = LaBraMEncoder(**encoder_config)
        checkpoint_sha256 = None
    probe_config = dict(channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST, num_classes=2, dropout=.1)
    model = LinearProbe(encoder, **probe_config)
    trainer = ClassifierTrainer(model, num_classes=2, lr=.001,
        accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-labram-training'))
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
        sampling_rate=200, window_seconds=4, chunk_size=800, patch_size=200, patches=4,
        scale_microvolts=100.,
        filters=dict(bandpass_hz=(.1, 75.), notch_hz=50., method='MNE FIR on full recordings after resampling'),
        pooling='learned native norm after raw patch-token mean excluding CLS',
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        allow_native_checkpoint=args.allow_native_checkpoint, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=metrics), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, artifact=str(output)))


if __name__ == '__main__':
    main()
