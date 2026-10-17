"""
Train EEGNet on EEGMat
========================================

Read the EDF files from the EEG During Mental Arithmetic Tasks dataset:
https://physionet.org/content/eegmat/1.0.0/.
Cite Zyma et al. (2019) and PhysioNet; the data use the Open Data Commons
Attribution License v1.0. Download at least two complete subject pairs.

This example distinguishes background from arithmetic, using four-second
windows, 19 scalp EEG channels, and 200 Hz sampling. Normalize each window
per channel. Hold out the last subject in sorted ID order for testing.
Train for a fixed number of epochs without using the test subject for
validation or model selection. This is a training example, not a published
benchmark score or the complete EEG-Arena protocol.

Install the checked-out source and run::

    pip install -e ".[dataset,trainer]"

    python examples/examples_eegmat_eegnet.py --root-path ./eegmat --epochs 1

Use a fresh cache path after changing source files or preprocessing. The
cache and split directories are separate and do not contain raw EDF files.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on EEGMat with one held-out subject.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-19ch-200hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')

    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        chunk_size=800,
        num_channel=19,
        sampling_rate=200,
        io_path=args.cache_path,
        io_mode='pickle',
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1),
            transforms.ToTensor(),
            transforms.To2d(),
        ]),
        label_transform=transforms.Select('label'),
        num_worker=0,
    )
    subjects = sorted(dataset.info['subject_id'].unique())
    if len(subjects) < 2:
        raise ValueError('Download at least two subjects with both background and arithmetic recordings.')
    for subject in subjects:
        labels = set(dataset.info.loc[dataset.info.subject_id == subject, 'label'])
        if labels != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')

    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train_dataset, test_dataset = next(splitter.split(dataset, subject=subjects[-1]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = EEGNet(num_electrodes=19, chunk_size=800, num_classes=2)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
                default_root_dir=str(Path(args.cache_path).parent / 'eegmat-training'),
                logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'metrics': result})


if __name__ == '__main__':
    main()
