"""
Continue symmetric TFM masked-token prediction on EEGMat
========================================================

Use real EEG During Mental Arithmetic Tasks recordings (Zyma et al., 2019):
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
Cite Pradeepkumar et al., TFM-Tokenizer, ICLR 2026. Author source and weights
are MIT; retain LICENSES/TFMTokenizer.txt and BIOT.txt. Supply the official
multi-dataset complete MLP tokenizer and8192-class MTP encoder; no random fallback/download.

Resample whole EDF records to 200 Hz, then FIR .1--75 Hz and notch 50 Hz.
Preserve the source reference. Four-second windows retain the first 16 of
19 measured electrodes, excluding Fz/Cz/Pz, scaled per channel by their own
absolute95th percentile+1e-8. Channel rows0--15 are sequential; no physical
mapping to CHB-MIT bipolar electrodes is claimed. Native STFT uses a periodic
float32 CPU Hann200 moved to the signal device, hop100, centerFalse, bins0--99.

Hold out the last sorted subject. Labels identify complete conditions only;
no labels enter self-supervised training. Use sequential batches wrapping at
the end, fixed AdamW1e-5/wd1e-5/betas(.9,.99). This is a few-step teaching
transfer, not the source's large corpus, cosine/warmup schedule or benchmark.
--steps is the total optimizer-step count, including resumed steps. Save full
model/EMA, AdamW, masking/dropout RNG, source hashes, protocol and next cursor.
Validate on the first four held-out windows with a separate fixed mask RNG;
no held-out update or model selection. Use a new output path and a new cache
when data or preprocessing changes. This workflow uses CPU.

Freeze every teacher parameter and keep teacher eval even during training.
Use the complete official8192-class MTP head, not the binary CHB-MIT head or a
random replacement. Two complementary token views use ID8192 for masking;
sum each partition's mean CE. No raw-EEG gradient passes through hard IDs.
A sampled empty partition is rejected explicitly rather than returning NaN.

Run::

    python examples/examples_eegmat_tfm_mtp_training.py --root-path ./eegmat \
        --tokenizer-checkpoint ./tfm_tokenizer_last.pth \
        --encoder-checkpoint ./tfm_encoder_mtp_last.pth --steps 2 --output-path ./mtp-step2.pt

    python examples/examples_eegmat_tfm_mtp_training.py --root-path ./eegmat \
        --resume ./mtp-step2.pt --steps 3 --output-path ./mtp-step3.pt
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
from torcheeg.models import TFMTokenizer, TFMEncoder, TFMMaskedTokenPretrainer


def filter_recording(eeg):
    eeg = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=200,
        l_freq=0.1,
        h_freq=75.0,
        verbose=False,
    )
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.0], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples.')
    selected = np.asarray(eeg[:16], dtype=np.float64)
    return (
        selected / (np.quantile(np.abs(selected), 0.95, axis=-1, keepdims=True) + 1e-8)
    ).astype(np.float32)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build_model(tokenizer_config, encoder_config, quantization_mode):
    tokenizer = TFMTokenizer(**tokenizer_config)
    return TFMMaskedTokenPretrainer(tokenizer, TFMEncoder(**encoder_config))


def train_step(model, optimizer, eeg, generator):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(eeg, generator=generator)
    if not torch.isfinite(output['loss']):
        raise ValueError('Nonfinite training loss.')
    output['loss'].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise ValueError(f'Missing or nonfinite gradient: {name}.')
    optimizer.step()
    return output['loss'].detach().item()


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
        description='Continue symmetric TFM masked-token prediction on EEGMat'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument(
        '--cache-path', default='./eegmat-tfm-pretraining-19ch-200hz-4s'
    )
    parser.add_argument('--tokenizer-checkpoint')
    parser.add_argument('--encoder-checkpoint')
    parser.add_argument('--resume')

    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--validation-windows', type=int, default=4)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    if (
        bool(args.tokenizer_checkpoint) == bool(args.resume)
        or min(args.steps, args.batch_size, args.validation_windows) < 1
        or (not args.resume and not args.encoder_checkpoint)
        or (args.resume and args.encoder_checkpoint)
    ):
        parser.error(
            'Supply a complete initial checkpoint pair or resume, and positive counts.'
        )
    if Path(args.output_path).exists():
        parser.error('output-path already exists; select a new output path.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    files = sorted(Path(args.root_path).glob('Subject*.edf'))
    if not files:
        raise ValueError('Supply real local EEGMat EDF recordings.')
    hashes = {path.name: sha256(path) for path in files}
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        io_path=args.cache_path,
        io_mode='pickle',
        num_channel=19,
        sampling_rate=200,
        chunk_size=800,
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_window), transforms.ToTensor()]
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
    quantization_mode = None
    protocol = dict(
        objective='mtp',
        sampling_rate=200,
        chunk_size=800,
        channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST[:16],
        channel_ids=list(range(16)),
        normalization='per-window/channel abs95percentile+1e-8; no population fit',
        filters='whole-record 200Hz resampling then .1--75Hz FIR and 50Hz notch; preserve source reference',
        stft='periodic float32 CPU Hann200, hop100, centerFalse, magnitude bins0--99',
        train_subjects=subjects[:-1],
        held_out_subject=subjects[-1],
        train_windows=len(train_rows),
        test_windows=len(test_rows),
        source_sha256=hashes,
        batch_size=args.batch_size,
        quantization_mode=quantization_mode,
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=1e-5, betas=[0.9, 0.99]),
    )
    generator = torch.Generator().manual_seed(43)
    if args.resume:
        payload = torch.load(args.resume, weights_only=True, map_location='cpu')
        if payload['protocol'] != protocol:
            raise ValueError(
                'Resume protocol, partition, batch size, objective or source files changed.'
            )
        tokenizer_config, encoder_config = (
            payload['tokenizer_config'],
            payload['encoder_config'],
        )
        model = build_model(tokenizer_config, encoder_config, quantization_mode)
        model.load_state_dict(payload['model'], strict=True)
        step, cursor = payload['step'], payload['cursor']
        initial_hashes = payload['initial_checkpoint_sha256']
        generator.set_state(payload['mask_generator'])
    else:
        tokenizer = TFMTokenizer.from_pretrained(
            args.tokenizer_checkpoint, reconstruction_head='mlp'
        )
        encoder = TFMEncoder.from_pretrained(args.encoder_checkpoint, num_classes=8192)
        model = TFMMaskedTokenPretrainer(tokenizer, encoder)
        tokenizer_config = dict(
            hid_channels=tokenizer.hid_channels,
            num_tokens=tokenizer.num_tokens,
            encoder_depth=tokenizer.encoder_depth,
            decoder_depth=tokenizer.decoder_depth,
            reconstruction_head=tokenizer.reconstruction_head,
            decay=tokenizer.decay,
            update_codebook=tokenizer.update_codebook,
            beta=tokenizer.beta,
        )
        encoder_config = dict(
            hid_channels=encoder.hid_channels,
            num_tokens=encoder.num_tokens,
            num_layers=encoder.num_layers,
            num_classes=encoder.num_classes,
        )
        step, cursor = 0, 0
        initial_hashes = dict(
            tokenizer=sha256(args.tokenizer_checkpoint),
            encoder=sha256(args.encoder_checkpoint),
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=1e-5,
        betas=(0.9, 0.99),
    )
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
        torch.random.set_rng_state(payload['torch_rng'])
    if args.steps <= step:
        raise ValueError('steps must exceed the saved optimizer-step count.')
    losses = []
    while step < args.steps:
        rows = train_rows[cursor : cursor + args.batch_size]
        eeg = torch.stack([dataset[index][0] for index in rows]).float()
        losses.append(train_step(model, optimizer, eeg, generator))
        cursor = (cursor + len(rows)) % len(train_rows)
        step += 1
    model.eval()
    validation_generator = torch.Generator().manual_seed(44)
    validation_losses = []
    with torch.no_grad():
        for index in test_rows[: args.validation_windows]:
            validation_losses.append(
                model(
                    dataset[index][0].unsqueeze(0).float(),
                    generator=validation_generator,
                )['loss'].item()
            )
    if hashes != {path.name: sha256(path) for path in files}:
        raise ValueError('Original source files changed during training.')
    save_checkpoint(
        args.output_path,
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            tokenizer_config=tokenizer_config,
            encoder_config=encoder_config,
            protocol=protocol,
            mask_generator=generator.get_state(),
            torch_rng=torch.random.get_rng_state(),
            step=step,
            cursor=cursor,
            initial_checkpoint_sha256=initial_hashes,
            train_losses_this_run=losses,
            validation_loss=float(np.mean(validation_losses)),
            validation_windows=len(validation_losses),
        ),
    )
    print(
        dict(
            objective='mtp',
            model_states=len(model.state_dict()),
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
