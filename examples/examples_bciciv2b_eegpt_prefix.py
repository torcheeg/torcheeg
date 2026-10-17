"""
Train EEGPT prefix on BCIC-IV-2B
================================

Read BCI Competition IV 2b through MOABB using C3, Cz, C4, a 4--38 Hz
bandpass, and 256 Hz resampling. Use 0--4 seconds relative to the imagery
interval (3--7 seconds after events), one 1024-sample window per retained
epoch. MOABB supplies microvolts. Standardize each channel within its
window. Use left_hand=0, right_hand=1. Hold out the last selected subject;
use fixed training epochs without model selection on its labels.

Cite Leeb et al. (2007), Tangermann et al. (2012), MOABB and Wang et al.
(2024), EEGPT, plus the tuning-method paper linked in the API reference.
Obtain recordings under their terms at
https://bnci-horizon-2020.eu/database/data-sets.

The default small encoder is RANDOMLY INITIALIZED: width32, two blocks,
four heads and two summary tokens. This run verifies the tuning pipeline;
it is not pretrained EEGPT validation or a published benchmark score.
Supply --checkpoint for a complete local official-large encoder state;
--checkpoint-prefix '' accepts a plain encoder tensor dictionary. No
checkpoint is downloaded automatically or loaded with unsafe pickle.

Freeze all base encoder weights and train only prefix plus the
classification head. Install the registered tuning modules before creating
the trainer/optimizer. Use freeze_encoder=False on the task model so the
new tuning modules enter training mode. Their settings are prefix_length=4.

Save the full tensor state, encoder/head/tuning settings and fixed seed.
To restore it, construct the encoder, task head and this tuning structure,
then load_state_dict with strict=True. The tutorial shows this sequence.

Run::

    python examples/examples_bciciv2b_eegpt_prefix.py --download-path ./bciciv2b-raw --epochs 1

Keep raw downloads outside processed caches. Use new cache/split paths
when subjects, sources or preprocessing change.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets.moabb import MOABBDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_prefix
from torcheeg.trainers import ClassifierTrainer


def label_to_index(label):
    return {'left_hand': 0, 'right_hand': 1}[label]


def main():
    parser = argparse.ArgumentParser(description='Train an EEGPT encoder and a linear head on BCIC-IV-2B with one held-out subject.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-eegpt-prefix-3ch-256hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint', help='Local complete official-large encoder checkpoint.')
    parser.add_argument('--checkpoint-prefix', default='target_encoder.')
    parser.add_argument('--output-path', default='./bciciv2b-eegpt-prefix.pt')
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
    model = EEGPTClassifier(encoder, num_classes=2)
    add_eegpt_prefix(model, prefix_length=4)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
        default_root_dir=str(Path(args.cache_path).parent / 'bciciv2b-eegpt-prefix-training'),
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    encoder_config = dict(parameters, hid_channels=encoder.hid_channels, depth=encoder.depth,
        num_heads=encoder.num_heads, embed_num=encoder.embed_num, patch_size=encoder.patch_size,
        patch_stride=encoder.patch_stride, norm_eps=encoder.norm.eps)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tuning_config = {'prefix_length': 4}
    torch.save({'state_dict': model.state_dict(), 'encoder_config': encoder_config,
        'head_config': {'num_classes': 2, 'dropout': 0.5, 'max_norm': 1.0},
        'tuning_method': 'prefix', 'tuning_config': tuning_config,
        'checkpoint_loaded': bool(args.checkpoint), 'seed': 42, 'epochs': args.epochs,
        'held_out_subject': subjects[-1], 'torch_version': str(torch.__version__)}, output)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
        'test_windows': len(test_dataset), 'metrics': result, 'checkpoint_loaded': bool(args.checkpoint),
        'tuning_method': 'prefix', 'trainable_parameters': trainable,
        'total_parameters': total, 'artifact': str(output)})


if __name__ == '__main__':
    main()
