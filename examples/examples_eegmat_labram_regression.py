"""
Regress EEGMat subtraction scores with LaBraMRegression
=======================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Jiang et al. (ICLR 2024), LaBraM. Source recordings use
Open Data Commons Attribution License v1.0. No data or weights are downloaded.

Supply at least three subjects and subject-info.csv. Predict the published
Number of subtractions from arithmetic windows only. Each subject shares
one behavioral target across their windows. Hold out the last sorted subject
and require two different training-subject targets. Divide targets by the
fixed factor 10, train with absolute error and report MAE in original units
by multiplying by 10. No held-out target selects epochs or parameters.

The EDF reader returns microvolts and resamples complete recordings to
200 Hz. Apply .1--75 Hz bandpass and 50 Hz notch MNE FIR to full recordings,
preserve all 19 source channels/reference/order, then divide microvolts by
100. Take four-second, 800-sample windows containing four native temporal
patches. No per-window z-score or population-fitted statistics are applied.
The wrapper selects whole-window resampling, an identity at this native rate.

Freeze the encoder in evaluation mode. Apply the complete native learned
final norm PER TOKEN and discard CLS, retaining channel-major patch order.
Train flatten -> 4H -> H -> outputs with ELU/dropout .1 after each hidden layer.
This preserves the archived all-token/flatten topology with complete native
norm restoration. It differs from norm-after-mean pooling or a freshly
initialized downstream fc_norm. H is the encoder width.

The default RANDOM encoder has width100/conv4/two blocks. Optional local
--checkpoint uses canonical base width200/conv8/12-block state. For the
author training file add --checkpoint-prefix student. and
--allow-native-checkpoint to explicitly enable known metadata and recognized
student pretraining-head removal with weights_only=True. Random outputs
are not pretrained features; this teaching protocol is not a published
benchmark or EEG-Arena reproduction.

Save complete tensor state, both constructor configurations, preprocessing,
channel/partition settings and all held-out predictions and targets.
Rebuild LaBraMEncoder and the task wrapper with the saved settings, then
strictly restore tensor state. No optimizer state is saved. Keep EEG-derived
artifacts under source data-use terms. Choose new cache/split paths when
recordings or offline preprocessing change.

Run::

    python examples/examples_eegmat_labram_regression.py --root-path ./eegmat --epochs 1
"""
import argparse
import hashlib
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import LaBraMEncoder, LaBraMRegression
from torcheeg.trainers import RegressorTrainer


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


def read_scores(path):
    table = pd.read_csv(path)
    required = ['Subject', 'Number of subtractions']
    if table.empty or any(name not in table.columns for name in required):
        raise ValueError('Metadata must contain Subject and Number of subtractions columns.')
    subjects = table['Subject'].astype(str).str.strip().str.lower()
    values = pd.to_numeric(table['Number of subtractions'], errors='raise')
    if (not subjects.str.fullmatch(r'subject[0-9]+').all() or subjects.duplicated().any()
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Metadata needs unique subject IDs and finite nonnegative scores.')
    return dict(zip(subjects, values.astype(float)))


def main():
    parser = argparse.ArgumentParser(description='Regress EEGMat subtraction scores with LaBraM.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--metadata-path')
    parser.add_argument('--cache-path', default='./eegmat-labram-regression-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-labram-regression.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.allow_native_checkpoint and not args.checkpoint:
        parser.error('native-checkpoint mode requires --checkpoint.')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    metadata_path = Path(args.metadata_path or Path(args.root_path) / 'subject-info.csv')
    scores = read_scores(metadata_path)
    metadata_sha256 = file_sha256(metadata_path)
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=800, sampling_rate=200, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Compose([transforms.Select('subject_id'),
            transforms.Lambda(lambda subject: scores[str(subject).lower()] / 10.)]))
    dataset.info = dataset.info.loc[dataset.info.label == 1].reset_index(drop=True)
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 3:
        raise ValueError('Use at least three subjects with arithmetic-condition recordings.')
    if any(str(subject).lower() not in scores for subject in subjects):
        raise ValueError('Every recorded subject must have a metadata score.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    if len({scores[str(subject).lower()] for subject in train.info.subject_id.unique()}) < 2:
        raise ValueError('Use at least two different training-subject targets.')
    backbone_config = dict(patch_size=200, hid_channels=200, conv_channels=8, mlp_channels=800,
        depth=12, num_heads=10, qkv_bias=False, qk_norm=True, dropout=0., attention_dropout=0.,
        drop_path_rate=0., layer_scale=.1, norm_eps=1e-6, use_abs_pos_emb=True,
        max_patches=16, interpolate_time=False, pooling='mean')
    if args.checkpoint:
        backbone = LaBraMEncoder.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix,
            allow_training_metadata=args.allow_native_checkpoint, exclude_pretraining=args.allow_native_checkpoint, **backbone_config)
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        backbone_config.update(hid_channels=100, conv_channels=4, mlp_channels=200, depth=2, num_heads=4)
        backbone = LaBraMEncoder(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST, chunk_size=800, sampling_rate=200,
        num_outputs=1, pooling='flatten', head_channels=None, dropout=.1, resample_mode='window', normalize_features=True, freeze_encoder=True)
    model = LaBraMRegression(backbone, **task_config)
    trainer = RegressorTrainer(model, lr=.001,
        accelerator='cpu', devices=1, metrics=['mae'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-labram-regression-training'))
    test_loader = DataLoader(test, batch_size=args.batch_size)
    metrics = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for eeg, target in test_loader:
            predictions.append(model(eeg).cpu())
            targets.append(target.cpu())
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), backbone_config=backbone_config, task_config=task_config,
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=200, native_sampling_rate=200, patch_size=200, patches=4, window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.1, 75.), notch_hz=50., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_predictions=torch.cat(predictions), test_targets=torch.cat(targets), metrics=metrics,
        metadata_sha256=metadata_sha256, target='Number of subtractions', target_scale=10., mae_metadata_units=metrics['test_mae'] * 10.), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, mae_metadata_units=metrics['test_mae'] * 10., artifact=str(output)))


if __name__ == '__main__':
    main()
