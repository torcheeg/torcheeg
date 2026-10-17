"""
Train EEGNet on PhysioNet motor imagery
========================================

Download EDF files from https://physionet.org/content/eegmmidb/1.0.0/.
Cite Schalk (2009), Schalk et al. (2004), and PhysioNet. The data use the
Open Data Commons Attribution License v1.0. For a small functional run,
download S001/S001R04.edf, S001/S001R06.edf, S002/S002R04.edf and
S002/S002R06.edf, keeping their subject directories.

Use average reference, a 0.3 Hz high-pass filter, a 60 Hz notch filter and
200 Hz resampling. Extract four-second imagery epochs in microvolts from
64 channels. Labels identify left fist, right fist, both fists and both
feet. Normalize each channel within each window; this does not fit any
statistics on other subjects. Hold out the last subject in sorted order.
Train for fixed epochs without test-set model selection. This small run
is a functional validation, not the full EEG-Arena evaluation protocol.

Install the checked-out source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_physionet_mi_eegnet.py --root-path ./eegmmidb --epochs 1

Use a new cache path after changing source files or preprocessing.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import PhysioNetMIDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on PhysioNet motor imagery with one held-out subject.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./physionet-mi-64ch-200hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')

    pl.seed_everything(42, workers=True)
    dataset = PhysioNetMIDataset(
        root_path=args.root_path,
        chunk_size=800,
        num_channel=64,
        sfreq=200,
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
        raise ValueError('Download at least two subjects with left/right and both-fists/feet imagery runs.')
    for subject in subjects:
        labels = set(dataset.info.loc[dataset.info.subject_id == subject, 'label'])
        if labels != {0, 1, 2, 3}:
            raise ValueError(f'Subject {subject} needs all four imagery classes (for example, runs R04 and R06).')

    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train_dataset, test_dataset = next(splitter.split(dataset, subject=subjects[-1]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = EEGNet(num_electrodes=64, chunk_size=800, num_classes=4)
    trainer = ClassifierTrainer(model=model, num_classes=4, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
                default_root_dir=str(Path(args.cache_path).parent / 'physionet-mi-training'),
                logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'metrics': result})


if __name__ == '__main__':
    main()
