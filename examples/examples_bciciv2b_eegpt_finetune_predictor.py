"""
Train the full EEGPT task path with its predictor
=================================================

Train EEGPTFinetuneClassifier: spatial encoder, temporal predictor,
patch-token mean pooling, LayerNorm and a constrained linear classifier.
The default run uses small random components (width 32, two blocks each,
two summaries). This checks training, not pretrained or benchmark scores.
Provide --checkpoint only for a complete local canonical-large encoder and
predictor checkpoint. The joint loader safely reads both complete tensor
components, validates all their keys and shapes, and initializes a new task
head. It never downloads weights or falls back to unsafe pickle loading.

BCIC-IV-2B uses bipolar C3, Cz, C4 in order. Cite Leeb et al. (2007),
Tangermann et al. (2012), MOABB and Wang et al. (2024), EEGPT. Obtain data
under their terms at https://bnci-horizon-2020.eu/database/data-sets.
MOABB supplies microvolts; use a 4--38 Hz bandpass, resample to 256 Hz, and
keep the first 1024 samples of the 0--4 second imagery interval (3--7 seconds
from the cue). Standardize each channel within its own window. Labels are
left_hand=0, right_hand=1. Hold out the last subject; never tune settings or
epochs on its labels. This is an explicit teaching protocol; it does not
reproduce the full EEG-Arena training schedule. One epoch checks functionality.

Save the complete task tensor state, constructor settings, preprocessing,
subject IDs and all held-out logits/targets with --output-path. Reload with
the same constructor configuration and weights_only=True, then evaluate in
model.eval() mode. The saved task state differs from a pretrained-component
checkpoint accepted by --checkpoint. Optional --freeze-encoder keeps its
dropout in evaluation mode while training the decoder and task head.

Install the checked-out library and run::

    pip install -e ".[dataset,trainer]"
    export MNE_DATASETS_BNCI_PATH="$PWD/bciciv2b-raw"
    python examples/examples_bciciv2b_eegpt_finetune_predictor.py --download-path "$MNE_DATASETS_BNCI_PATH" --epochs 1

Set the BNCI environment variable and --download-path to the same directory:
current MOABB reads that variable again when extracting epochs. Keep raw
files outside the processed cache. Use a new cache/split path when changing
source, subjects or preprocessing.
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
from torcheeg.models import EEGPT, EEGPTPredictor, EEGPTFinetuneClassifier
from torcheeg.trainers import ClassifierTrainer



def label_to_index(label):
    return {'left_hand': 0, 'right_hand': 1}[label]


def main():
    parser = argparse.ArgumentParser(description='Train the full EEGPT predictor task path on BCIC-IV-2B with one held-out subject.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-eegpt-finetune-predictor-3ch-256hz-4s')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint', help='Complete local canonical-large encoder and predictor checkpoint.')
    parser.add_argument('--encoder-prefix', default='target_encoder.')
    parser.add_argument('--decoder-prefix', default='predictor.')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--output-path', default='./bciciv2b-eegpt-finetune-predictor.pt')
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
    encoder_config = dict(num_electrodes=3, chunk_size=1024, channel_names=['C3', 'CZ', 'C4'])
    decoder_config = dict(num_patches=16)
    task_config = dict(num_classes=2, freeze_encoder=args.freeze_encoder)
    if args.checkpoint:
        model = EEGPTFinetuneClassifier.from_pretrained(args.checkpoint, encoder_config, decoder_config,
            decoder_type='predictor', encoder_prefix=args.encoder_prefix,
            decoder_prefix=args.decoder_prefix, **task_config)
    else:
        encoder_config.update(hid_channels=32, depth=2, num_heads=4, embed_num=2)
        decoder_config.update(in_features=32, hid_channels=32, depth=2, num_heads=4, embed_num=2)
        model = EEGPTFinetuneClassifier(EEGPT(**encoder_config), EEGPTPredictor(**decoder_config), **task_config)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=1e-3,
                                accelerator='cpu', devices=1, metrics=['accuracy'])
    trainer.fit(train_loader, None, max_epochs=args.epochs,
        default_root_dir=str(Path(args.cache_path).parent / 'bciciv2b-eegpt-finetune-predictor-training'),
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    result = trainer.test(test_loader, logger=False, enable_checkpointing=False)[0]
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in test_loader:
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(task_state=model.state_dict(), encoder_config=encoder_config,
        decoder_config=decoder_config, task_config=task_config, decoder_type='predictor',
        checkpoint_loaded=bool(args.checkpoint), sampling_rate=256, units='microvolts',
        preprocessing=dict(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
            tmin=0, tmax=4, samples=1024, normalization='within-window per-channel mean/std'),
        train_subjects=subjects[:-1], test_subject=subjects[-1],
        test_logits=torch.cat(logits), test_targets=torch.cat(targets), metrics=result), output)
    print({'held_out_subject': subjects[-1], 'train_windows': len(train_dataset),
           'test_windows': len(test_dataset), 'decoder': 'predictor',
           'checkpoint_loaded': bool(args.checkpoint), 'encoder_frozen': args.freeze_encoder,
           'saved_model_path': str(output), 'metrics': result})


if __name__ == '__main__':
    main()
