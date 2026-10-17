"""
Train DeepConvNet on BCI Competition IV 2b with MOABB
=====================================================

Read the BNCI2014_004 dataset through MOABB. Cite Leeb et al. (2007),
Tangermann et al. (2012), and MOABB. Obtain and use the source data under
the dataset's terms: https://bnci-horizon-2020.eu/database/data-sets.
The script does not distribute recordings or trained weights. Cite
Schirrmeister et al. (2017), DOI: 10.1002/hbm.23730, for DeepConvNet.

Use the bipolar C3, Cz, C4 EEG channels. MOABB reads the 250 Hz recordings,
applies a 4--38 Hz bandpass, and resamples to 128 Hz. MOABB's numeric arrays
use microvolts for this dataset. Select 0--4 seconds relative to the start
of the dataset's imagery interval (3--7 seconds relative to its events),
then use one 512-sample window per retained epoch. Normalize each channel
within its window. Map left_hand to 0 and right_hand to 1.

The model uses the EEG-Arena architecture settings: no temporal bias,
convolution max-norm 2.0, classifier max-norm 0.5, and native PyTorch weight
initialization. The temporal kernel contains 10 samples; each max-pooling
kernel and stride contains 3 samples. Classifier dropout is 0.5. Kernel adaptation is enabled
explicitly for inputs too short for the requested shared kernel. These
lengths refer to the 128 Hz input.

Hold out the last selected subject. Train for a fixed number of epochs
without validation or hyperparameter selection on that subject. This small
training workflow is not the complete EEG-Arena protocol or a benchmark score.

Install MOABB and the checked-out library, then run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_bciciv2b_deepconvnet.py --download-path ./bciciv2b-raw --epochs 1

The explicit download path is registered with MOABB and must be outside the
processed cache. Downloads reuse existing source files. Use a fresh cache and
split path when changing sources, subjects, or preprocessing.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets.moabb import MOABBDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import DeepConvNet
from torcheeg.trainers import ClassifierTrainer


def label_to_index(label):
    return {'left_hand': 0, 'right_hand': 1}[label]


def main():
    parser = argparse.ArgumentParser(description='Train DeepConvNet on BCIC-IV-2B with one held-out subject.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-deep-3ch-128hz-4s')
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
                                tmin=0, tmax=4, resample=128)
    dataset = MOABBDataset(dataset=source, paradigm=paradigm,
        download_path=args.download_path, io_path=args.cache_path, io_mode='pickle',
        chunk_size=512, overlap=0, num_worker=0,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
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
    model = DeepConvNet(num_electrodes=3, chunk_size=512, num_classes=2,
        temporal_bias=False, conv_max_norm=2.0, classifier_max_norm=0.5,
        initialization='pytorch', classifier_dropout=0.5, auto_kernel=True)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
        default_root_dir=str(Path(args.cache_path).parent / 'bciciv2b-deep-training'),
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'metrics': result})


if __name__ == '__main__':
    main()
