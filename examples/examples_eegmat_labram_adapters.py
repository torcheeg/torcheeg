"""
Train LaBraM adapters on subject-held-out EEGMat
================================================

Use public EEG During Mental Arithmetic Tasks recordings:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Jiang et al. (ICLR 2024), LaBraM. Source recordings use
Open Data Commons Attribution License v1.0. Obtain two complete subject
pairs. Classify background=0 and arithmetic=1, holding out the last sorted
subject. Fix epochs and do not select settings from held-out labels.

The reader returns microvolts and resamples full recordings to 200 Hz.
Apply .1--75 Hz bandpass and 50 Hz notch MNE FIR to complete recordings,
preserve all 19 source channels/reference/order and divide microvolts by
100. Use four-second windows with four complete 200-sample patches. Select
whole-window resampling, an identity at this native rate. No per-window
z-score or fitted population statistics are applied.

The task uses native learned final norm PER TOKEN, then channel-major
flatten -> 4H -> H -> 2 with ELU/dropout .1. It differs from norm-after-mean
pooling or a fresh native fc_norm. The default RANDOM encoder has width100,
conv4 and two blocks. Optional local --checkpoint uses canonical base
width200/conv8/12-block state. For the verified author training file add
--checkpoint-prefix student. and --allow-native-checkpoint to explicitly
allow known metadata and recognized student pretraining-head removal with
weights_only=True. No data or weights are downloaded.

Install two residual bottleneck adapters per block, after its complete
attention and MLP residuals, including native layer scales. Zero-initialized
up-projections preserve the initial deterministic base function. Freeze all
native encoder parameters and train only adapters and the task head.
freeze_encoder=False permits registered adapter dropout (.1) to train;
freeze_base=True freezes base parameters without changing module modes.
The base encoder's own dropout and stochastic depth are zero here. Create
the trainer/optimizer after installing adapters.

Save complete native/adapter/head tensor state, all three constructor
configurations, preprocessing/channel/partition settings and every held-out
logit/target. To restore, rebuild LaBraMEncoder and LaBraMClassifier, install
adapters with the saved adapter_config, then load state strictly. No optimizer
state is saved or resumed. This teaching run does not reproduce published
LaBraM/adapter/EEG-Arena scores. Choose new cache/split paths when sources or
offline preprocessing change.

Run::

    python examples/examples_eegmat_labram_adapters.py --root-path ./eegmat --epochs 1
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
from torcheeg.models import LaBraMEncoder, LaBraMClassifier, add_labram_adapters
from torcheeg.trainers import ClassifierTrainer


def filter_recording(eeg):
    eeg = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64), sfreq=200, l_freq=0.1, h_freq=75.0, verbose=False
    )
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.0], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.0


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Train LaBraMClassifier on subject-held-out EEGMat.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-labram-adapters-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-labram-adapters.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--adapter-size', type=int, default=16)
    args = parser.parse_args()
    if args.allow_native_checkpoint and not args.checkpoint:
        parser.error('native-checkpoint mode requires --checkpoint.')
    if args.epochs < 1 or args.batch_size < 1 or args.adapter_size < 1:
        parser.error('epochs, batch-size and adapter-size must be positive.')
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        io_path=args.cache_path,
        io_mode='pickle',
        num_channel=19,
        chunk_size=800,
        sampling_rate=200,
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(scale_microvolts), transforms.ToTensor()]
        ),
        label_transform=transforms.Select('label'),
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Use at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(
        LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    backbone_config = dict(
        patch_size=200,
        hid_channels=200,
        conv_channels=8,
        mlp_channels=800,
        depth=12,
        num_heads=10,
        qkv_bias=False,
        qk_norm=True,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_scale=0.1,
        norm_eps=1e-6,
        use_abs_pos_emb=True,
        max_patches=16,
        interpolate_time=False,
        pooling='mean',
    )
    if args.checkpoint:
        backbone = LaBraMEncoder.from_pretrained(
            args.checkpoint,
            prefix=args.checkpoint_prefix,
            allow_training_metadata=args.allow_native_checkpoint,
            exclude_pretraining=args.allow_native_checkpoint,
            **backbone_config,
        )
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        backbone_config.update(
            hid_channels=100, conv_channels=4, mlp_channels=200, depth=2, num_heads=4
        )
        backbone = LaBraMEncoder(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(
        channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST,
        chunk_size=800,
        sampling_rate=200,
        num_classes=2,
        pooling='flatten',
        head_channels=None,
        dropout=0.1,
        resample_mode='window',
        normalize_features=True,
        freeze_encoder=False,
    )
    model = LaBraMClassifier(backbone, **task_config)
    adapter_config = dict(
        adapter_size=args.adapter_size, dropout=0.1, blocks=None, freeze_base=True
    )
    add_labram_adapters(model, **adapter_config)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = ClassifierTrainer(
        model, num_classes=2, lr=0.001, accelerator='cpu', devices=1, metrics=['accuracy']
    )
    trainer.fit(
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        None,
        max_epochs=args.epochs,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-labram-adapters-training'),
    )
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
    torch.save(
        dict(
            state_dict=model.state_dict(),
            backbone_config=backbone_config,
            task_config=task_config,
            adapter_config=adapter_config,
            trainable_parameters=trainable_parameters,
            source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
            channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
            sampling_rate=200,
            native_sampling_rate=200,
            patch_size=200,
            patches=4,
            window_seconds=4,
            scale_microvolts=100.0,
            filters=dict(
                bandpass_hz=(0.1, 75.0),
                notch_hz=50.0,
                method='MNE FIR on full recordings after resampling',
            ),
            checkpoint_loaded=bool(args.checkpoint),
            checkpoint_sha256=checkpoint_sha256,
            held_out_subject=subjects[-1],
            train_subjects=subjects[:-1],
            seed=42,
            epochs=args.epochs,
            test_logits=torch.cat(logits),
            test_targets=torch.cat(targets),
            metrics=metrics,
        ),
        output,
    )
    print(
        dict(
            held_out_subject=subjects[-1],
            train_windows=len(train),
            test_windows=len(test),
            checkpoint_loaded=bool(args.checkpoint),
            checkpoint_sha256=checkpoint_sha256,
            trainable_parameters=trainable_parameters,
            metrics=metrics,
            artifact=str(output),
        )
    )


if __name__ == '__main__':
    main()
