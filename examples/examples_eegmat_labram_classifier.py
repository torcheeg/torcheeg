"""
Train LaBraMClassifier on subject-held-out EEGMat
=================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Jiang et al. (ICLR 2024), LaBraM. Source recordings use
Open Data Commons Attribution License v1.0. No data or weights are downloaded.

Supply at least two complete subject pairs. Classify background=0 and
arithmetic=1, holding out the last sorted subject. Fix epochs and do not
select parameters or epochs from held-out labels.

The EDF reader returns microvolts and resamples complete recordings to
200 Hz. Apply .1--75 Hz bandpass and 50 Hz notch MNE FIR to full recordings,
preserve all 19 source channels/reference/order, then divide microvolts by
100. Take four-second, 800-sample windows containing four native temporal
patches. No per-window z-score or population-fitted statistics are applied.
The wrapper selects whole-window resampling, an identity at this native rate.

Freeze the encoder in evaluation mode. Apply the complete native learned
final norm PER TOKEN and discard CLS, retaining channel-major patch order.
Train flatten -> 4H -> H -> outputs with ELU/dropout .1 after each hidden layer.
This preserves the archived all-token/flatten topology with complete native
norm restoration. It differs from norm-after-mean pooling or a freshly
initialized downstream fc_norm. H is the encoder width.

The default RANDOM encoder has width100/conv4/two blocks. Optional local
--checkpoint uses canonical base width200/conv8/12-block state. For the
author training file add --checkpoint-prefix student. and
--allow-native-checkpoint to explicitly enable known metadata and recognized
student pretraining-head removal with weights_only=True. Random outputs
are not pretrained features; this teaching protocol is not a published
benchmark or EEG-Arena reproduction.

Save complete tensor state, both constructor configurations, training parameter
names/count, learning rate, preprocessing,
channel/partition settings and all held-out predictions and targets.
Rebuild LaBraMEncoder and the task wrapper with the saved settings, then
strictly restore tensor state. No optimizer state is saved. Keep EEG-derived
artifacts under source data-use terms. Choose new cache/split paths when
recordings or offline preprocessing change.

Run::

    python examples/examples_eegmat_labram_classifier.py --root-path ./eegmat --epochs 1
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
from torcheeg.models import LaBraMEncoder, LaBraMClassifier
from torcheeg.trainers import ClassifierTrainer


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.1, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Train LaBraMClassifier on subject-held-out EEGMat.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-labram-classifier-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-labram-classifier.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.allow_native_checkpoint and not args.checkpoint:
        parser.error('native-checkpoint mode requires --checkpoint.')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=800, sampling_rate=200, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Use at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(dataset, subject=subjects[-1]))
    backbone_config = dict(patch_size=200, hid_channels=200, conv_channels=8, mlp_channels=800,
        depth=12, num_heads=10, qkv_bias=False, qk_norm=True, dropout=0., attention_dropout=0.,
        drop_path_rate=0., layer_scale=.1, norm_eps=1e-6, use_abs_pos_emb=True,
        max_patches=16, interpolate_time=False, pooling='mean')
    if args.checkpoint:
        backbone = LaBraMEncoder.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix,
            allow_training_metadata=args.allow_native_checkpoint, exclude_pretraining=args.allow_native_checkpoint, **backbone_config)
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        backbone_config.update(hid_channels=100, conv_channels=4, mlp_channels=200, depth=2, num_heads=4)
        backbone = LaBraMEncoder(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST, chunk_size=800, sampling_rate=200,
        num_classes=2, pooling='flatten', head_channels=None, dropout=.1, resample_mode='window', normalize_features=True, freeze_encoder=True)
    model = LaBraMClassifier(backbone, **task_config)
    trainable_parameter_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = ClassifierTrainer(model, num_classes=2, lr=.001,
        accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-labram-classifier-training'))
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
    torch.save(dict(fine_tuning='head', learning_rate=.001, trainable_parameters=trainable_parameters,
        trainable_parameter_names=trainable_parameter_names, state_dict=model.state_dict(), backbone_config=backbone_config, task_config=task_config,
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=200, native_sampling_rate=200, patch_size=200, patches=4, window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.1, 75.), notch_hz=50., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], seed=42, epochs=args.epochs,
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=metrics), output)
    print(dict(held_out_subject=subjects[-1], train_windows=len(train), test_windows=len(test),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        metrics=metrics, artifact=str(output)))


if __name__ == '__main__':
    main()
