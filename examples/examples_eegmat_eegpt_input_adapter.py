"""
Train an EEGPT input adapter on EEGMat
======================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Wang et al. (2024), EEGPT. Source data use the Open Data
Commons Attribution License v1.0. Download at least two complete subject
pairs. Classify background=0 versus mental arithmetic=1.

Read the dataset's fixed 19-channel order. Its EDF reader supplies
microvolts and resamples complete recordings to 200 Hz. Take 800-sample
four-second windows and standardize each channel within its window.
The differentiable input adapter resamples each whole window to 1024
samples, representing the same four seconds at 256 Hz, then learns three
feature slots with channel mixing and a depthwise temporal kernel of 15
samples. Its Fourier method treats windows as periodic; boundaries can ring.

The encoder token IDs C3/CZ/C4 label LEARNED SLOTS. The adapter does not
recover actual C3/Cz/C4 sensor waveforms. Source channel order must remain
fixed between training and evaluation. The saved artifact records this
source order and the learned-slot IDs separately.

Freeze the encoder and train the adapter plus the classification head.
The default encoder is SMALL AND RANDOMLY INITIALIZED: width32, two
blocks, four heads and two summary tokens. A one-epoch run tests the
training path, not pretrained EEGPT performance. Optional --checkpoint
loads a complete local official-large encoder state with the strict safe
loader; use --checkpoint-prefix '' for a plain encoder tensor dictionary.
No weights are downloaded or loaded with unsafe deserialization.

Hold out the last subject in sorted ID order, with no test-label tuning.
Train normalization statistics on training windows; evaluation uses eval
mode. Save the complete tensor state and constructor settings. Restore
adapter, encoder, head and EEGPTAdaptedModel before strict state loading.

Run::

    python examples/examples_eegmat_eegpt_input_adapter.py --root-path ./eegmat --epochs 1

Use a new processed cache and split path when sources or preprocessing
change. The script does not download or redistribute recordings.
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
from torcheeg.models import EEGPT, EEGPTClassifier, EEGPTInputAdapter, EEGPTAdaptedModel
from torcheeg.trainers import ClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train an EEGPT input adapter through a frozen encoder.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-eegpt-input-19ch-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-eegpt-input-adapter.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='target_encoder.')
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
    parameters = dict(num_electrodes=3, chunk_size=1024, channel_names=['C3', 'CZ', 'C4'])
    if args.checkpoint:
        encoder = EEGPT.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **parameters)
    else:
        encoder = EEGPT(hid_channels=32, depth=2, num_heads=4, embed_num=2, **parameters)
    adapter_config = dict(in_channels=19, out_channels=3, num_samples=1024,
                          temporal_kernel=15, dropout=.25, max_norm=1.0)
    head_config = dict(num_classes=2, dropout=.5, max_norm=1.0, freeze_encoder=True)
    model = EEGPTAdaptedModel(EEGPTInputAdapter(**adapter_config), EEGPTClassifier(encoder, **head_config))
    trainer = ClassifierTrainer(model, num_classes=2, accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(DataLoader(train, batch_size=args.batch_size, shuffle=True), None,
        max_epochs=args.epochs, logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        default_root_dir=str(Path(args.cache_path).parent / 'eegmat-eegpt-input-training'))
    metrics = trainer.test(DataLoader(test, batch_size=args.batch_size),
                           logger=False, enable_checkpointing=False)[0]
    encoder_config = dict(parameters, hid_channels=encoder.hid_channels, depth=encoder.depth,
        num_heads=encoder.num_heads, embed_num=encoder.embed_num, patch_size=encoder.patch_size,
        patch_stride=encoder.patch_stride, norm_eps=encoder.norm.eps)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'adapter_config': adapter_config,
        'encoder_config': encoder_config, 'head_config': head_config,
        'source_channel_order': MENTAL_ARITHMETIC_CHANNEL_LIST,
        'encoder_token_names': parameters['channel_names'],
        'source_sampling_rate': 200, 'target_sampling_rate': 256, 'window_seconds': 4,
        'checkpoint_loaded': bool(args.checkpoint), 'seed': 42, 'epochs': args.epochs,
        'held_out_subject': subjects[-1], 'torch_version': str(torch.__version__)}, output)
    print({'held_out_subject': subjects[-1], 'train_windows': len(train), 'test_windows': len(test),
           'metrics': metrics, 'checkpoint_loaded': bool(args.checkpoint),
           'source_samples': 800, 'adapted_samples': 1024, 'artifact': str(output)})


if __name__ == '__main__':
    main()
