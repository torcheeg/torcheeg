"""
Regress EEGMat subtraction scores with CBraModRegression
========================================================

Use the public EEG During Mental Arithmetic Tasks database and its
subject-info.csv: https://physionet.org/content/eegmat/1.0.0/.
Cite Zyma et al. (2019), PhysioNet and Wang et al. (2025), CBraMod.
Source data use the Open Data Commons Attribution License v1.0.
Obtain at least three complete subject pairs and the metadata CSV.

Predict the published Number of subtractions column from arithmetic
condition windows only. A subject's windows share that behavioral target.
Hold out the last sorted subject, requiring two different targets among
training subjects. Divide targets by the fixed factor 10; no target
statistics or population preprocessing are fitted. Train with absolute
error and report MAE in metadata units by multiplying normalized MAE by 10.
No held-out target is used to select parameters or epochs.

The reader supplies microvolts and resamples full recordings to 256 Hz.
Filter each complete recording at .3--75 Hz and notch 60 Hz with MNE FIR,
retain the source EEG reference, reorder its 19 channels to the author's
order and divide microvolts by 100. Take four-second windows. The model
resamples EACH of four one-second segments from 256 to 200 samples using
Fourier interpolation. This assumes periodic segments and can cause ringing;
it differs from the author's full-recording preprocessing at 200 Hz.

Freeze the complete backbone in evaluation mode and train a flatten ->
4H -> H -> 1 head with ELU/dropout .1 after each hidden layer. The default
32-wide, two-block backbone is RANDOM. Optional --checkpoint loads the
complete canonical 200-wide, 12-block local author state safely. Nothing is
downloaded. This teaching protocol is not a validated behavioral predictor
or a pretrained benchmark. Keep subject isolation and fixed epochs.

Save complete tensor state, constructor/preprocessing settings and all
held-out predictions/targets. Rebuild CBraModRegression with those settings
before strict state loading. Use new cache/split paths for changed sources
or preprocessing.

Run::

    python examples/examples_eegmat_cbramod_regression.py --root-path ./eegmat --epochs 1
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
from torcheeg.models import CBraMod, CBraModRegression
from torcheeg.trainers import RegressorTrainer


CHANNEL_ORDER = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=256,
        l_freq=.3, h_freq=75, verbose=False)
    return mne.filter.notch_filter(eeg, Fs=256, freqs=[60.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


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
    parser = argparse.ArgumentParser(description='Regress EEGMat subtraction scores with CBraMod.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--metadata-path')
    parser.add_argument('--cache-path', default='./eegmat-cbramod-regression-19ch-256hz-4s')
    parser.add_argument('--output-path', default='./eegmat-cbramod-regression.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    scores = read_scores(args.metadata_path or Path(args.root_path) / 'subject-info.csv')
    pl.seed_everything(42, workers=True)
    indices = [MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=1024, sampling_rate=256, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.PickElectrode(indices),
            transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
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
    backbone_config = dict(patch_size=200, out_features=200, hid_channels=200, mlp_channels=800,
        depth=12, num_heads=8, dropout=.1, norm_eps=1e-5, position_kernel=(19, 7))
    if args.checkpoint:
        backbone = CBraMod.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **backbone_config)
        checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    else:
        backbone_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        backbone = CBraMod(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(num_electrodes=19, chunk_size=1024, num_patches=4, num_outputs=1,
                       dropout=.1, pooling='flatten', head_channels=None, freeze_encoder=True)
    model = CBraModRegression(backbone, **task_config)
    trainer = RegressorTrainer(model, lr=.001,
        accelerator='cpu', devices=1, metrics=['mae'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-cbramod-regression-training'))
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
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=CHANNEL_ORDER,
        sampling_rate=256, target_patch_sampling_rate=200, patch_size=200, patches=4, window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.3, 75.), notch_hz=60., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_predictions=torch.cat(predictions), test_targets=torch.cat(targets), metrics=metrics,
        target='Number of subtractions', target_scale=10., mae_metadata_units=metrics['test_mae'] * 10.), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, mae_metadata_units=metrics['test_mae'] * 10., artifact=str(output)))


if __name__ == '__main__':
    main()
