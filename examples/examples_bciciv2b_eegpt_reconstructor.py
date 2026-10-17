"""
Train EEGPT with its temporal reconstructor on BCIC-IV-2B
====================================================================

Use the official EEGPT full-token reconstructor after an encoder, followed
by patch-token mean pooling, LayerNorm and an unconstrained linear head.
This small randomly initialized teaching pipeline is not a pretrained model
or the full EEG-Arena task wrapper. It implements no masked pretraining loss.
Unused native mask tensors are retained for state compatibility.

Read BCIC-IV-2B with MOABB. Cite Leeb et al. (2007), Tangermann et al.
(2012), MOABB and Wang et al. (2024), EEGPT. Obtain the source recordings
under their terms at https://bnci-horizon-2020.eu/database/data-sets.
Use bipolar C3, Cz, C4 in that order, 4--38 Hz bandpass and 256 Hz resampling.
MOABB supplies microvolts. Keep the 0--4 second imagery interval (3--7 seconds
from the cue event), taking 1024 samples. Standardize each channel within its
own window. Labels are left_hand=0, right_hand=1. Hold out the last subject;
never use that subject to select epochs, preprocessing or hyperparameters.

The encoder and temporal decoder have width 32 and two blocks. Train them
jointly; the decoder receives [batch, 16 patches, 2 summaries, 32 features].
The class token is excluded from patch-token mean pooling. A one-epoch run
checks training and evaluation; its accuracy is not a benchmark score.

Install the checked-out library and run::

    pip install -e ".[dataset,trainer]"
    export MNE_DATASETS_BNCI_PATH="$PWD/bciciv2b-raw"
    python examples/examples_bciciv2b_eegpt_reconstructor.py --download-path "$MNE_DATASETS_BNCI_PATH" --epochs 1

MOABB BNCI reads the named environment variable again during epoch loading;
set it to the same raw directory to reuse downloads across both stages.
Keep raw data outside the processed cache. Select a new cache and split path
when changing subjects, sources or preprocessing.
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
from torcheeg.models import EEGPT, EEGPTReconstructor
from torcheeg.trainers import ClassifierTrainer


class PatchMeanPool(nn.Module):
    def forward(self, tokens):
        return tokens[:, 1:].mean(dim=1)


def label_to_index(label):
    return {'left_hand': 0, 'right_hand': 1}[label]


def main():
    parser = argparse.ArgumentParser(description='Train an EEGPT encoder and temporal reconstructor on BCIC-IV-2B with one held-out subject.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-eegpt-reconstructor-3ch-256hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
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
    encoder = EEGPT(hid_channels=32, depth=2, num_heads=4, embed_num=2, **parameters)
    decoder = EEGPTReconstructor(num_patches=encoder.num_patches, in_features=32,
        embed_num=2, hid_channels=32, depth=2, num_heads=4)
    model = nn.Sequential(encoder, decoder, PatchMeanPool(), nn.LayerNorm(32), nn.Linear(32, 2))
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
        default_root_dir=str(Path(args.cache_path).parent / 'bciciv2b-eegpt-reconstructor-training'),
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'decoder': 'reconstructor', 'random_initialization': True, 'metrics': result})


if __name__ == '__main__':
    main()
