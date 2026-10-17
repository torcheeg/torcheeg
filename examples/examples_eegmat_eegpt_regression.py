"""
Regress EEGMat subtraction scores with EEGPTRegression
======================================================

Use the public EEG During Mental Arithmetic Tasks database and its
subject-info.csv: https://physionet.org/content/eegmat/1.0.0/.
Cite Zyma et al. (2019), PhysioNet, and Wang et al. (2024), EEGPT.
Source data use the Open Data Commons Attribution License v1.0.
Download at least three complete subject pairs and the metadata CSV.

Predict the published Number of subtractions column, using arithmetic
condition windows only. All windows of one subject share that subject's
behavioral target. Hold out the last subject in sorted ID order, so no
participant occurs in both training and testing. Train for fixed epochs;
do not tune parameters on the held-out target. This small teaching protocol
is not a published EEGPT benchmark or a claim of generalizable performance.

Use the dataset's 19-channel order, explicit historical electrode aliases
T3=T7, T4=T8, T5=P7, T6=P8, microvolt arrays, 256 Hz resampling, and
four-second/1024-sample windows. Standardize each channel within its window.
Scale targets by the fixed factor 10, without fitting target statistics.
The trainer uses absolute-error loss. Report MAE in the original metadata
units by multiplying its normalized-target MAE by 10.

Train a small RANDOMLY INITIALIZED encoder (width32, two blocks, four heads,
two summary tokens) and a scalar constrained regression head. No pretrained
weights, population preprocessing or hyperparameter search are used.

Run::

    python examples/examples_eegmat_eegpt_regression.py --root-path ./eegmat --epochs 1

Use a new processed cache when changing raw sources or preprocessing. This
script does not download or redistribute recordings or train on test labels.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGPT, EEGPTRegression
from torcheeg.trainers import RegressorTrainer


def read_scores(path):
    table = pd.read_csv(path)
    required = ['Subject', 'Number of subtractions']
    if any(name not in table.columns for name in required):
        raise ValueError('Metadata must contain Subject and Number of subtractions columns.')
    subjects = table['Subject'].astype(str).str.strip().str.lower()
    values = pd.to_numeric(table['Number of subtractions'], errors='raise')
    if (not subjects.str.fullmatch(r'subject[0-9]+').all() or subjects.duplicated().any()
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Metadata needs unique subject IDs and finite nonnegative scores.')
    return dict(zip(subjects, values.astype(float)))


def main():
    parser = argparse.ArgumentParser(description='Train EEGPTRegression on EEGMat behavioral scores.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--metadata-path')
    parser.add_argument('--cache-path', default='./eegmat-eegpt-score-19ch-256hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    scores = read_scores(args.metadata_path or Path(args.root_path) / 'subject-info.csv')
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', chunk_size=1024, num_channel=19, sampling_rate=256,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()]),
        label_transform=transforms.Compose([transforms.Select('subject_id'),
            transforms.Lambda(lambda subject: scores[str(subject).lower()] / 10.0)]), num_worker=0)
    dataset.info = dataset.info.loc[dataset.info.label == 1].reset_index(drop=True)
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 3:
        raise ValueError('Use at least three subjects with arithmetic-condition recordings.')
    if any(str(subject).lower() not in scores for subject in subjects):
        raise ValueError('Every recorded subject must have a metadata score.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train, test = next(splitter.split(dataset, subject=subjects[-1]))
    if len({scores[str(subject).lower()] for subject in train.info.subject_id.unique()}) < 2:
        raise ValueError('Use at least two different training-subject targets.')
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    channels = [aliases.get(name.upper(), name.upper()) for name in MENTAL_ARITHMETIC_CHANNEL_LIST]
    encoder = EEGPT(num_electrodes=19, chunk_size=1024, channel_names=channels,
                    hid_channels=32, depth=2, num_heads=4, embed_num=2)
    model = EEGPTRegression(encoder)
    trainer = RegressorTrainer(model, accelerator='cpu', devices=1, metrics=['mae'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-eegpt-regression-training'))
    metrics = trainer.test(DataLoader(test, batch_size=args.batch_size),
                           logger=False, enable_checkpointing=False)[0]
    mae = metrics['test_mae'] * 10.0
    if not np.isfinite(mae):
        raise RuntimeError('Regression produced a nonfinite MAE.')
    print({'held_out_subject': subjects[-1], 'train_windows': len(train), 'test_windows': len(test),
           'target': 'Number of subtractions', 'target_scale': 10.0,
           'mae_metadata_units': mae, 'checkpoint_loaded': False})


if __name__ == '__main__':
    main()
