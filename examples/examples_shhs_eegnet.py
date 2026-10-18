"""
Train EEGNet on SHHS
===================

Obtain access to SHHS EDF and NSRR XML files from
https://sleepdata.org/datasets/shhs. Cite Quan et al. (1997) and NSRR.
Keep the provider's polysomnography directory structure.

Use EEG and left EOG, a 0.5--30 Hz FIR filter, 100 Hz output and complete
30-second epochs. Normalize each epoch per channel. Hold out the final
participant in sorted identifier order, including both visits when present.
Train for a fixed epoch count without selecting settings on the test set.
This is a training workflow, not a published sleep-staging benchmark.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_shhs_eegnet.py --root-path ./SHHS --epochs 1

At least two participants are required. For a short functional check, add
``--end-at 180``; early-night cropped records can contain only wake labels,
so such a check does not measure five-class sleep-staging performance.
Use a new cache path whenever files, visits, crop or preprocessing change.
The documentation gallery does not execute this training script.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import SHHSDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on SHHS with one held-out participant.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./shhs-eeg-eog-100hz-30s')
    parser.add_argument('--visits', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--start-at', type=float, default=0.)
    parser.add_argument('--end-at', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')

    pl.seed_everything(42, workers=True)
    dataset = SHHSDataset(
        root_path=args.root_path, visits=args.visits, channels=['EEG', 'EOG(L)'],
        sfreq=100, l_freq=0.5, h_freq=30., start_at=args.start_at, end_at=args.end_at,
        io_path=args.cache_path, io_mode='pickle', num_worker=0,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Mapping({'Sleep stage W': 0, 'Sleep stage N1': 1,
                                'Sleep stage N2': 2, 'Sleep stage N3': 3,
                                'Sleep stage R': 4})]))
    subjects = sorted(dataset.info['subject_id'].unique())
    if len(subjects) < 2:
        raise ValueError('At least two SHHS participants with complete EDF/NSRR pairs are required.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train_dataset, test_dataset = next(splitter.split(dataset, subject=subjects[-1]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = EEGNet(num_electrodes=2, chunk_size=3000, num_classes=5)
    trainer = ClassifierTrainer(model=model, num_classes=5, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
                default_root_dir=str(Path(args.cache_path).parent / 'shhs-training'),
                logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_epochs': len(train_dataset),
           'test_epochs': len(test_dataset), 'visits': args.visits,
           'start_at': args.start_at, 'end_at': args.end_at, 'metrics': result})


if __name__ == '__main__':
    main()
