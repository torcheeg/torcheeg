"""
Train a complete EEGPT task with input adaptation and LoRA
=========================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Wang et al. (2024), EEGPT. Source data use the Open Data
Commons Attribution License v1.0. Obtain at least two complete subject
pairs. Classify background=0 versus mental arithmetic=1.

The fixed pipeline demonstrates composable library components. Read all
19 source channels in their dataset order, in microvolts at 200 Hz. Extract
800-sample four-second windows and standardize each channel within its
window. The Fourier input adapter maps the whole window to 1024 samples
(same duration at 256 Hz), then learns five features. The full task model
maps those five features to three encoder slots, encodes spatial summaries,
runs its temporal reconstructor, pools and classifies. Fourier windows are
periodic and can ring at boundaries. The learned features and encoder token
IDs C3/CZ/C4 do not recover physical C3/Cz/C4 measurements. These successive
learned mappings are explicit; applications can instead select a direct
mapping with matching task input channels.

Install registered LoRA updates in the spatial encoder before creating the
optimizer. Freeze its original weights, train its LoRA parameters, and train
the input adapter, task channel mapper, decoder, pooling norm and head. All
components start from RANDOM weights; width 32 and two blocks keep this
teaching run inexpensive. It verifies composition, not pretrained accuracy
or an EEG-Arena training protocol. No pretrained checkpoint is downloaded.

Hold out the last subject in sorted ID order; do not tune on its labels.
Only training batches update adapter normalization statistics; evaluate in
model.eval() mode. Save complete constructor settings and state, held-out
logits and targets. Restore encoder, decoder and task, install the same
LoRA modules, wrap the input adapter, then safely load the full tensor state
with strict=True. The source channel order and learned slot IDs are saved
separately.

Run::

    python examples/examples_eegmat_eegpt_composition.py --root-path ./eegmat --epochs 1

The script does not download or redistribute source files. Keep raw data
outside the processed cache, and use a new cache/split path when changing
sources or preprocessing. A one-epoch result is a functional check.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import (EEGPT, EEGPTReconstructor, EEGPTFinetuneClassifier,
    EEGPTInputAdapter, EEGPTAdaptedModel, add_eegpt_lora)
from torcheeg.trainers import ClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train a complete EEGPT task with an input adapter and LoRA.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-eegpt-composition-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-eegpt-composition.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    pl.seed_everything(42, workers=True)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', chunk_size=800, num_channel=19, sampling_rate=200,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()]),
        label_transform=transforms.Select('label'), num_worker=0)
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Use at least two subjects with both recording conditions.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs background and arithmetic recordings.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train, test = next(splitter.split(dataset, subject=subjects[-1]))
    encoder_config = dict(num_electrodes=3, chunk_size=1024, channel_names=['C3','CZ','C4'],
                          hid_channels=32, depth=2, num_heads=4, embed_num=2)
    decoder_config = dict(num_patches=16, in_features=32, hid_channels=32,
                          depth=2, num_heads=4, embed_num=2)
    task_config = dict(num_classes=2, in_channels=5)
    tuning_config = dict(rank=2, alpha=4., dropout=.1, freeze_base=True)
    adapter_config = dict(in_channels=19, out_channels=5, num_samples=1024,
                          temporal_kernel=15, dropout=.25, max_norm=1.)
    task = EEGPTFinetuneClassifier(EEGPT(**encoder_config),
        EEGPTReconstructor(**decoder_config), **task_config)
    add_eegpt_lora(task, **tuning_config)
    model = EEGPTAdaptedModel(EEGPTInputAdapter(**adapter_config),task)
    trainer = ClassifierTrainer(model, num_classes=2, accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-eegpt-composition-training'))
    metrics = trainer.test(DataLoader(test, batch_size=args.batch_size),
                           logger=False, enable_checkpointing=False)[0]
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in DataLoader(test,batch_size=args.batch_size):
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'adapter_config': adapter_config,
        'encoder_config': encoder_config, 'decoder_config': decoder_config,
        'task_config': task_config, 'tuning_config': tuning_config,
        'source_channel_order': MENTAL_ARITHMETIC_CHANNEL_LIST,
        'encoder_token_names': encoder_config['channel_names'],
        'source_sampling_rate': 200, 'target_sampling_rate': 256, 'window_seconds': 4,
        'checkpoint_loaded': False, 'seed': 42, 'epochs': args.epochs,
        'held_out_subject': subjects[-1], 'torch_version': str(torch.__version__),
        'test_logits': torch.cat(logits), 'test_targets': torch.cat(targets), 'metrics': metrics}, output)
    print({'held_out_subject': subjects[-1], 'train_windows': len(train), 'test_windows': len(test),
           'metrics': metrics, 'checkpoint_loaded': False, 'task_input_channels': 5,
           'encoder_slots': 3, 'source_samples': 800, 'adapted_samples': 1024, 'artifact': str(output)})


if __name__ == '__main__':
    main()
