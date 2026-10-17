"""
Train an EEGPT encoder and linear head on BCIC-IV-2B
===================================================

Read BCI Competition IV 2b with MOABB. Cite Leeb et al. (2007), Tangermann
et al. (2012), MOABB, and Wang et al. (2024), EEGPT. Obtain source recordings
under the dataset terms at https://bnci-horizon-2020.eu/database/data-sets.

Use bipolar C3, Cz, C4 in this order, a 4--38 Hz bandpass and 256 Hz
resampling. MOABB supplies microvolts. Retain 0--4 seconds relative to its
imagery interval (3--7 seconds after events) and take 1024 samples per epoch.
Normalize each channel within each window. Use left_hand=0 and right_hand=1.
Hold out the last subject; do not tune parameters or epochs on its labels.

By default this script trains a SMALL RANDOMLY INITIALIZED encoder, with
width 32, two blocks, four heads and two summary tokens. It verifies the
training path, not pretrained EEGPT performance. The default EEGPT library
architecture has width 512, eight blocks and four summary tokens. Provide
--checkpoint to load that architecture from a complete local official-large
encoder state. The loader checks every encoder key and shape and uses safe
weights-only loading. It rejects unsupported pickle objects instead of
falling back to unsafe loading. No weights are downloaded automatically.
Use --checkpoint-prefix '' for a plain encoder tensor dictionary; the
default prefix is target_encoder. inside a Lightning state_dict.

Optional --freeze-encoder trains only the linear head. No resampling or
channel projection happens inside the encoder. These preprocessing and
training choices are an explicit teaching protocol, not the full published
EEGPT or EEG-Arena evaluation protocol. One epoch is a functional check.

Install the checked-out library and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_bciciv2b_eegpt.py --download-path ./bciciv2b-raw --epochs 1

Keep raw downloads outside the processed cache. A new cache and split path
is required when changing subjects, sources or preprocessing.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets.moabb import MOABBDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGPT
from torcheeg.trainers import ClassifierTrainer


def label_to_index(label):
    return {'left_hand': 0, 'right_hand': 1}[label]


def main():
    parser = argparse.ArgumentParser(description='Train an EEGPT encoder and a linear head on BCIC-IV-2B with one held-out subject.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-eegpt-3ch-256hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint', help='Local complete official-large encoder checkpoint.')
    parser.add_argument('--checkpoint-prefix', default='target_encoder.')
    parser.add_argument('--freeze-encoder', action='store_true')
    args = parser.parse_args()
    subjects = sorted(set(args.subjects))
    if len(subjects) < 2 or any(subject < 1 or subject > 9 for subject in subjects):
        parser.error('Select at least two different subject IDs between 1 and 9.')
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')

    pl.seed_everything(42, workers=True)
    source = BNCI2014_004()
    source.subject_list = subjects
    paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                tmin=0, tmax=4, resample=256)
    dataset = MOABBDataset(dataset=source, paradigm=paradigm,
        download_path=args.download_path, io_path=args.cache_path, io_mode='pickle',
        chunk_size=1024, overlap=0, num_worker=0,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()]),
        label_transform=transforms.Compose([
            transforms.Select('label'), transforms.Lambda(label_to_index)]))
    if set(dataset.info.subject_id) != set(subjects):
        raise ValueError('The processed source must contain every selected subject.')
    for subject in subjects:
        labels = set(dataset.info.loc[dataset.info.subject_id == subject, 'label'])
        if labels != {'left_hand', 'right_hand'}:
            raise ValueError(f'Subject {subject} must contain both imagery classes.')

    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train_dataset, test_dataset = next(splitter.split(dataset, subject=subjects[-1]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    parameters = dict(num_electrodes=3, chunk_size=1024, channel_names=['C3', 'CZ', 'C4'])
    if args.checkpoint:
        encoder = EEGPT.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **parameters)
    else:
        # A small, randomly initialized encoder keeps this teaching run inexpensive.
        encoder = EEGPT(hid_channels=32, depth=2, num_heads=4, embed_num=2, **parameters)
    if args.freeze_encoder:
        encoder.requires_grad_(False)
    model = nn.Sequential(encoder, nn.Flatten(start_dim=1),
                          nn.Linear(encoder.num_patches * encoder.embed_num * encoder.hid_channels, 2))
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
        default_root_dir=str(Path(args.cache_path).parent / 'bciciv2b-eegpt-training'),
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'checkpoint_loaded': bool(args.checkpoint),
           'encoder_frozen': args.freeze_encoder, 'metrics': result})


if __name__ == '__main__':
    main()
