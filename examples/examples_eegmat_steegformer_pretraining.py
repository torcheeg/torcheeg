"""
Continue the full ST-EEGFormer MAE on EEGMat
============================================

Supply local official Small, Base or Large weights and EEG During Mental
Arithmetic Tasks recordings: https://physionet.org/content/eegmat/1.0.0/.
Cite Zyma et al. (2019), PhysioNet, and Yang et al., ST-EEGFormer. Data use
Open Data Commons Attribution License v1.0; model source uses MIT.

This standalone teaching workflow trains every encoder and decoder parameter.
It requires complete pretrained weights; it never substitutes a small random
network. Use --allow-native-checkpoint for the official Small/Base files'
argparse.Namespace metadata. All loading remains weights_only=True.

Resample whole recordings to 128 Hz, then apply .5--45 Hz MNE FIR filtering.
Preserve all 19 channels and their recorded reference. Split into one-second
128-sample windows and standardize each channel using that window's population
standard deviation (floor 1e-6 microvolts). Use exact learned channel IDs,
16-sample patches, native position-0 encoder/position-1 decoder, mask ratio .75,
and unnormalized reconstruction targets. This differs from the author's large
pretraining corpus and EEG-Arena protocols; no benchmark score is claimed.

Hold out the last sorted subject. Do not train on held-out windows or labels.
Training takes sequential batches, wrapping at the end, without shuffling.
--steps is the total desired optimizer-step count, including resumed steps.
Save full model/AdamW states, independent mask-generator state, next batch
cursor, protocol, subject partition, original EDF hashes and validation loss.
Validation uses the first --validation-windows held-out windows and a separate
fixed mask generator; it does not affect the training generator or selection.

Run::

    python examples/examples_eegmat_steegformer_pretraining.py \
        --root-path ./eegmat --checkpoint ./checkpoint-300.pth \
        --allow-native-checkpoint --steps 2 --output-path ./mae-step2.pt

    python examples/examples_eegmat_steegformer_pretraining.py \
        --root-path ./eegmat --resume ./mae-step2.pt \
        --steps 3 --output-path ./mae-step3.pt

Use a fresh cache when raw data or preprocessing changes. Complete Large
training needs substantially more RAM than its 1.3 GB weights alone. The
workflow uses CPU and does not claim GPU or distributed-training validation.
"""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import mne
import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import STEEGFormerPretrainer


def filter_recording(eeg):
    return mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=128,
        l_freq=0.5,
        h_freq=45.0,
        verbose=False,
    )


def normalize_window(eeg):
    return (eeg - eeg.mean(-1, keepdims=True)) / np.maximum(
        eeg.std(-1, keepdims=True, ddof=0), 1e-6
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_step(model, optimizer, eeg, indices, generator):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = model(eeg, indices, mask_ratio=0.75, generator=generator)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite pretraining loss.')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f'Missing or nonfinite pretraining gradient: {name}.')
    optimizer.step()
    return loss.detach().item()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(
        description='Continue complete ST-EEGFormer MAE pretraining on EEGMat.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument(
        '--cache-path', default='./eegmat-steegformer-mae-19ch-128hz-1s'
    )
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument(
        '--variant', choices=('small', 'base', 'large'), default='small'
    )
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--validation-windows', type=int, default=4)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    if (
        bool(args.checkpoint) == bool(args.resume)
        or min(args.steps, args.batch_size, args.validation_windows) < 1
        or (args.allow_native_checkpoint and not args.checkpoint)
    ):
        parser.error(
            'Supply exactly one checkpoint/resume; counts must be positive; native metadata opt-in needs checkpoint.'
        )
    if Path(args.output_path).exists():
        parser.error('output-path already exists; select a new checkpoint path.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    raw_files = sorted(Path(args.root_path).glob('*.edf'))
    if not raw_files:
        raise ValueError('No local EEGMat EDF recordings found.')
    source_hashes = {p.name: file_sha256(p) for p in raw_files}
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        io_path=args.cache_path,
        io_mode='pickle',
        num_channel=19,
        chunk_size=128,
        sampling_rate=128,
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(normalize_window), transforms.ToTensor()]
        ),
        label_transform=transforms.Select('label'),
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Supply at least two complete subject pairs.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train_rows = dataset.info.index[dataset.info.subject_id != subjects[-1]].tolist()
    test_rows = dataset.info.index[dataset.info.subject_id == subjects[-1]].tolist()
    protocol = dict(
        sampling_rate=128,
        chunk_size=128,
        patch_size=16,
        bandpass_hz=[0.5, 45.0],
        normalization='per-window per-channel population z-score; floor 1e-6 uV',
        channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST,
        encoder_position_offset=0,
        norm_pix_loss=False,
        mask_ratio=0.75,
        train_subjects=subjects[:-1],
        held_out_subject=subjects[-1],
        train_windows=len(train_rows),
        test_windows=len(test_rows),
        batch_size=args.batch_size,
        source_sha256=source_hashes,
        variant=args.variant,
    )
    generator = torch.Generator().manual_seed(43)
    if args.resume:
        payload = torch.load(
            args.resume, map_location='cpu', weights_only=True, mmap=True
        )
        if payload['protocol'] != protocol:
            raise ValueError(
                'Resume protocol, partition, batch size or original recordings changed.'
            )
        model = STEEGFormerPretrainer.from_pretrained(args.resume, variant=args.variant)
        step, cursor = payload['step'], payload['cursor']
        generator.set_state(payload['mask_generator'])
        torch.random.set_rng_state(payload['torch_rng'])
        checkpoint_sha256 = payload['initial_checkpoint_sha256']
    else:
        model = STEEGFormerPretrainer.from_pretrained(
            args.checkpoint,
            variant=args.variant,
            allow_namespace=args.allow_native_checkpoint,
        )
        step, cursor = 0, 0
        checkpoint_sha256 = file_sha256(args.checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
    if args.steps <= step:
        raise ValueError('steps must exceed the saved optimizer-step count.')
    indices = model.channel_indices(MENTAL_ARITHMETIC_CHANNEL_LIST)
    losses = []
    while step < args.steps:
        batch_rows = train_rows[cursor : cursor + args.batch_size]
        eeg = torch.stack([dataset[row][0] for row in batch_rows]).float()
        losses.append(train_step(model, optimizer, eeg, indices, generator))
        cursor = (cursor + len(batch_rows)) % len(train_rows)
        step += 1
    model.eval()
    validation_losses = []
    validation_generator = torch.Generator().manual_seed(44)
    with torch.no_grad():
        for row in test_rows[: args.validation_windows]:
            eeg = dataset[row][0].unsqueeze(0).float()
            validation_losses.append(
                model(eeg, indices, generator=validation_generator)[0].item()
            )
    save_checkpoint(
        args.output_path,
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            mask_generator=generator.get_state(),
            torch_rng=torch.random.get_rng_state(),
            step=step,
            cursor=cursor,
            protocol=protocol,
            initial_checkpoint_sha256=checkpoint_sha256,
            train_losses_this_run=losses,
            validation_loss=float(np.mean(validation_losses)),
            validation_windows=len(validation_losses),
        ),
    )
    print(
        dict(
            variant=args.variant,
            full_model_states=len(model.state_dict()),
            step=step,
            cursor=cursor,
            train_losses_this_run=losses,
            held_out_subject=subjects[-1],
            validation_loss=float(np.mean(validation_losses)),
            validation_windows=len(validation_losses),
            checkpoint=str(Path(args.output_path).resolve()),
        )
    )


if __name__ == '__main__':
    main()
