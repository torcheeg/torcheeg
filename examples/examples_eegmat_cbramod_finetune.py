"""
Fine-tune CBraMod on EEGMat
===========================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Wang et al. (2025), CBraMod. Source recordings use the Open
Data Commons Attribution License v1.0. Obtain two complete subject pairs.
Classify background=0 versus arithmetic=1 and hold out the last sorted
subject. Do not tune on held-out labels.

The reader supplies microvolts and resamples complete recordings to 256 Hz.
Filter complete recordings at 0.3--75 Hz and notch 60 Hz using MNE FIR.
Reorder all 19 channels to the author's pretraining order, preserving the
source EEG reference, and divide microvolts by 100. Take four-second,
1024-sample windows. The model divides each into four one-second segments
and Fourier resamples EACH segment from 256 to 200 samples. Independent
patch resampling assumes periodic patches and can ring at their boundaries.
This differs from filtering/resampling full recordings directly at 200 Hz
and does not reproduce the pretraining or EEG-Arena benchmark protocol.

Train the native backbone and the flatten -> 4H -> H -> classes task head.
Use freeze_encoder=False without blindly enabling every tensor: the native
mask_encoding is fixed and must remain nontrainable. The native reconstruction
projection remains in the complete state, but is not used by this task and
receives no gradient. Dropout follows normal train/eval calls. GroupNorm and
LayerNorm have no running BatchNorm statistics.

A small RANDOM 32-wide, two-block backbone is the default. Optional
--checkpoint safely loads the complete local canonical 200-wide, 12-block
author tensor state before selecting trainable parameters. No data or weights
are downloaded. A one-epoch run checks functionality, not benchmark quality.
Create the optimizer after configuring trainable parameters.

Save complete tensor state, backbone/task/tuning configurations, trainable
parameter names, preprocessing, channel orders and all held-out logits and
targets. Rebuild the model with the same settings and parameter selection
before loading state strictly. Tensor state does not store requires_grad.
No optimizer state is saved; this artifact restores model behavior, not an
interrupted optimizer trajectory.

Run::

    python examples/examples_eegmat_cbramod_finetune.py --root-path ./eegmat --epochs 1

Select new cache/split paths when sources or offline preprocessing change.
"""
import argparse
import hashlib
from pathlib import Path

import mne
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import CBraMod, CBraModClassifier
from torcheeg.trainers import ClassifierTrainer


CHANNEL_ORDER = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=256,
        l_freq=.3, h_freq=75, verbose=False)
    return mne.filter.notch_filter(eeg, Fs=256, freqs=[60.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CBraMod on EEGMat.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-cbramod-finetune-19ch-256hz-4s')
    parser.add_argument('--output-path', default='./eegmat-cbramod-finetune.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    pl.seed_everything(42, workers=True)
    indices = [MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=1024, sampling_rate=256, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.PickElectrode(indices),
            transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Use at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    backbone_config = dict(patch_size=200, out_features=200, hid_channels=200, mlp_channels=800,
        depth=12, num_heads=8, dropout=.1, norm_eps=1e-5, position_kernel=(19, 7))
    if args.checkpoint:
        backbone = CBraMod.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **backbone_config)
        checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    else:
        backbone_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        backbone = CBraMod(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(num_electrodes=19, chunk_size=1024, num_patches=4, num_classes=2,
                       dropout=.1, pooling='flatten', head_channels=None, freeze_encoder=False)
    model = CBraModClassifier(backbone, **task_config)
    tuning_config = dict(strategy='full')
    trainer = ClassifierTrainer(model, num_classes=2, lr=.001,
        accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-cbramod-finetune-training'))
    test_loader = DataLoader(test, batch_size=args.batch_size)
    metrics = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in test_loader:
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), backbone_config=backbone_config, task_config=task_config, tuning_config=tuning_config,
        trainable_parameters=[name for name, parameter in model.named_parameters() if parameter.requires_grad],
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=CHANNEL_ORDER,
        sampling_rate=256, target_patch_sampling_rate=200, patch_size=200, patches=4, window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.3, 75.), notch_hz=60., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=metrics), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, artifact=str(output)))


if __name__ == '__main__':
    main()
