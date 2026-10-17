"""
Train a VAE on EEGMat feature grids
===================================

Train BEncoder and BDecoder with BetaVAETrainer, using the public EEG During
Mental Arithmetic Tasks data: https://physionet.org/content/eegmat/1.0.0/.
Cite Zyma et al. (2019) and PhysioNet. The data use the Open Data Commons
Attribution License v1.0. Download at least two complete subject pairs.

This example generates differential-entropy feature grids, not EEG waveforms.
Use 19 scalp channels in microvolts, 200 Hz sampling and four-second windows.
Compute theta/alpha/beta/gamma differential entropy, standardize each band
across the electrodes within that window, and map it to the dataset's 5 by 5
grid. Empty grid cells are zero. No population statistics are fitted.

Hold out the last subject in sorted ID order. Labels are returned by the
dataset but are not used by this unconditional VAE. Train for a fixed number
of epochs without validation or model selection on the test subject. Report
reconstruction MSE and the trainer's batch-summed KL divergence; these are
functional diagnostics, not a published benchmark or proof of EEG quality.
No pretrained classifier, image FID/IS, NNI or search service is required.

Run::

    python examples/examples_eegmat_bvae.py --root-path ./eegmat --epochs 1

Use a fresh cache after changing source files or preprocessing. The saved
artifact contains model states, configuration and four prior feature grids.
It does not contain source EDF files.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import BDecoder, BEncoder
from torcheeg.trainers import BetaVAETrainer


def feature_transform():
    return transforms.Compose([
        transforms.BandDifferentialEntropy(sampling_rate=200),
        transforms.MeanStdNormalize(axis=0),
        transforms.ToGrid(MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT),
    ])


def train_and_evaluate(train_dataset, test_dataset, epochs=10, batch_size=16):
    encoder = BEncoder(in_channels=4, grid_size=(5, 5), hid_channels=32)
    decoder = BDecoder(in_channels=32, out_channels=4, grid_size=(5, 5))
    trainer = BetaVAETrainer(encoder, decoder, beta=1e-3, lr=1e-3,
                             accelerator='cpu', devices=1, metrics=[])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    trainer.fit(train_loader, None, max_epochs=epochs, logger=False,
                enable_checkpointing=False, num_sanity_val_steps=0)
    metrics = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    trainer.eval()
    with torch.no_grad():
        generated = trainer(torch.randn(4, 32)).cpu()
    if not torch.isfinite(generated).all() or not all(
            torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise RuntimeError('Training produced non-finite outputs or metrics.')
    return trainer, generated, metrics


def main():
    parser = argparse.ArgumentParser(description='Train an unconditional EEGMat feature-grid VAE.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-bvae-de-5x5-4s')
    parser.add_argument('--output-path', default='./eegmat-bvae.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')

    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle',
        chunk_size=800, num_channel=19, sampling_rate=200,
        offline_transform=feature_transform(), online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'), num_worker=0)
    subjects = sorted(dataset.info['subject_id'].unique())
    if len(subjects) < 2:
        raise ValueError('Download at least two subjects with both recording conditions.')
    for subject in subjects:
        labels = set(dataset.info.loc[dataset.info.subject_id == subject, 'label'])
        if labels != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train_dataset, test_dataset = next(splitter.split(dataset, subject=subjects[-1]))
    trainer, generated, metrics = train_and_evaluate(
        train_dataset, test_dataset, args.epochs, args.batch_size)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'encoder': trainer.encoder.state_dict(),
                'decoder': trainer.decoder.state_dict(), 'generated_features': generated,
                'grid_size': (5, 5), 'latent_channels': 32, 'beta': trainer.beta,
                'epochs': args.epochs, 'held_out_subject': subjects[-1]}, output_path)
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'metrics': metrics,
           'generated_shape': list(generated.shape), 'artifact': str(output_path)})


if __name__ == '__main__':
    main()
