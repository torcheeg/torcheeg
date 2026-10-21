"""
Continue complete NeuroRVQ tokenizer training on EEGMat
=======================================================

Supply trusted author source at 27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c
and ntinosbarmpas/NeuroRVQ revision d944b87f44ae0ba2923b2f10d0518f23f6803b76,
pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt. Both are separately
licensed CC-BY-NC-4.0; neither is downloaded or bundled by this script.
Cite Barmpas et al., NeuroRVQ, and Zyma et al. (2019), EEGMat,
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Retain the complete 12-block encoder, 3-block decoder, four eight-layer RVQs,
and amplitude/sine/cosine heads. The objective is the original sum of code,
amplitude, phase and standardized-waveform losses, including native EMA.
Whole EDF records are resampled 500->200 Hz and FIR .5--45 Hz/notch50 filtered.
Keep all19 electrodes and source reference; each 4-second window/channel is
population-z-scored (std floor1e-6). This teaching transfer is not a published
pretraining protocol or benchmark. Output waveforms are standardized per sample
across all channels and times, not physical voltages.

Hold out the final sorted subject. No labels enter training. Sequential batches
wrap at the end. Fixed CPU AdamW1e-5/wd1e-5; --steps is TOTAL including resume.
Save every native state, optimizer, CPU RNG, metadata, source hashes and cursor.
Validate on the first four held-out windows; never update codebooks or usage
counters in evaluation. Use a new output file; use a new cache if inputs change.
No CUDA, distributed or compiled execution is selected by this example.

Run::

    python examples/examples_eegmat_neurorvq_tokenizer_training.py \
        --root-path ./eegmat --backend-root ./NeuroRVQ \
        --checkpoint ./NeuroRVQ_EEG_tokenizer_v1.pt --steps 2 --output-path ./step2.pt

    python examples/examples_eegmat_neurorvq_tokenizer_training.py \
        --root-path ./eegmat --backend-root ./NeuroRVQ \
        --resume ./step2.pt --steps 3 --output-path ./step3.pt
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import mne
import numpy as np
import torch
import yaml

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import NeuroRVQTokenizer

SOURCE_REVISION = '27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c'


def build_neurorvq_tokenizer(root):
    """Build all native tokenizer modules from explicitly trusted pinned source."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Require the documented pinned author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    sys.path.insert(0, str(root))
    from NeuroRVQ_EEG import NeuroRVQ as native
    from NeuroRVQ_EEG.NeuroRVQ_modules import get_encoder_decoder_params
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
    )

    if Path(native.__file__).resolve() != root / 'NeuroRVQ_EEG' / 'NeuroRVQ.py':
        raise ValueError(
            'Another author checkout is already imported; use a fresh process.'
        )
    vocab = [name.decode() for name in ch_names_global]
    config = yaml.safe_load((root / 'flags' / 'NeuroRVQ_EEG_v1.yml').read_text())
    config['n_global_electrodes'] = len(vocab)
    encoder, decoder = get_encoder_decoder_params(config)
    backend = native.NeuroRVQTokenizer(
        encoder,
        decoder,
        config['n_code'],
        config['code_dim'],
        config['decoder_out_dim'],
    )
    return backend, vocab


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=200,
        l_freq=0.5,
        h_freq=45.0,
        verbose=False,
    )
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.0], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite channels and 800 samples at200Hz.')
    eeg = np.asarray(eeg, dtype=np.float64)
    return (
        (eeg - eeg.mean(-1, keepdims=True)) / eeg.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cache_protocol(cache_path, source_hashes):
    """Reject stale or unverified caches; return the expected manifest."""
    path = Path(cache_path)
    marker = path / 'neurorvq-preprocessing.json'
    expected = dict(
        source_sha256=source_hashes,
        sampling_rate=200,
        window_samples=800,
        channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        filters='whole-record FIR.5--45Hz/notch50Hz after resampling',
        mne_version=str(mne.__version__),
    )
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError(
                'Cache input hashes or preprocessing differs; use a new cache path.'
            )
    elif path.exists() and any(path.iterdir()):
        raise ValueError(
            'Existing cache has no verified manifest; use a new cache path.'
        )
    return marker, expected


def main():
    parser = argparse.ArgumentParser(
        description='Continue full NeuroRVQ tokenizer on real EEGMat, CPU.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--cache-path', default='./eegmat-neurorvq-tokenizer-200hz-4s')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--validation-windows', type=int, default=4)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    if (
        bool(args.checkpoint) == bool(args.resume)
        or min(args.steps, args.batch_size, args.validation_windows) < 1
    ):
        parser.error('Supply an initial checkpoint OR resume, and positive counts.')
    if Path(args.output_path).exists():
        parser.error('output-path already exists; select a new file.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    files = sorted(Path(args.root_path).glob('Subject*.edf'))
    if not files:
        raise ValueError('Supply real local EEGMat EDF recordings.')
    hashes = {path.name: sha256(path) for path in files}
    marker, preprocessing = cache_protocol(args.cache_path, hashes)
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        chunk_size=800,
        num_channel=19,
        sampling_rate=200,
        io_path=args.cache_path,
        io_mode='pickle',
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_window), transforms.ToTensor()]
        ),
        label_transform=transforms.Select('label'),
    )
    if not marker.exists():
        marker.write_text(json.dumps(preprocessing, indent=2) + '\n')
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2 or any(
        set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}
        for subject in subjects
    ):
        raise ValueError('Supply at least two complete subject pairs.')
    train_rows = dataset.info.index[dataset.info.subject_id != subjects[-1]].tolist()
    test_rows = dataset.info.index[dataset.info.subject_id == subjects[-1]].tolist()
    backend, vocab = build_neurorvq_tokenizer(args.backend_root)
    config = dict(
        channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST,
        channel_vocab=vocab,
        sampling_rate=200,
        track_usage=False,
    )
    model = NeuroRVQTokenizer(backend, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    protocol = dict(
        source_sha256=hashes,
        source_revision=SOURCE_REVISION,
        config=config,
        train_rows=train_rows,
        test_rows=test_rows,
        batch_size=args.batch_size,
        filters='resample200Hz then whole-record FIR.5--45Hz/notch50Hz',
        normalization='per-window/channel population z-score, std floor1e-6',
        input_unit='microvolts before normalization',
        reference='source EDF, unchanged',
        window_samples=800,
        seed=42,
        mne_version=str(mne.__version__),
        optimizer=dict(name='AdamW', lr=1e-5, weight_decay=1e-5),
        weights_license='CC-BY-NC-4.0',
        held_out_subject=subjects[-1],
    )
    cursor, completed = 0, 0
    if args.resume:
        payload = torch.load(args.resume, weights_only=True, map_location='cpu')
        if payload['protocol'] != protocol:
            raise ValueError(
                'Resume input hashes, electrode order or protocol differs.'
            )
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        cursor, completed = payload['cursor'], payload['completed_steps']
        torch.set_rng_state(payload['rng_state'])
        initial_hash = payload['initial_checkpoint_sha256']
    else:
        model = NeuroRVQTokenizer.from_pretrained(args.checkpoint, backend, **config)
        initial_hash = sha256(args.checkpoint)
    if args.steps <= completed:
        raise ValueError('--steps must exceed completed steps.')
    losses = []
    for _ in range(completed, args.steps):
        selected = [
            train_rows[(cursor + i) % len(train_rows)] for i in range(args.batch_size)
        ]
        cursor = (cursor + args.batch_size) % len(train_rows)
        eeg = torch.stack([dataset[index][0] for index in selected])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_pretraining(eeg)
        if not torch.isfinite(result['loss']):
            raise ValueError('Nonfinite training loss.')
        result['loss'].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite gradient: ' + name)
        optimizer.step()
        losses.append(result['loss'].item())
    model.eval()
    evaluation = []
    with torch.no_grad():
        for index in test_rows[: args.validation_windows]:
            result = model.forward_pretraining(dataset[index][0].unsqueeze(0))
            evaluation.append(
                {
                    key: result[key].item()
                    for key in [
                        'loss',
                        'code_loss',
                        'amplitude_loss',
                        'phase_loss',
                        'signal_loss',
                    ]
                }
            )
    if hashes != {
        path.name: sha256(path)
        for path in sorted(Path(args.root_path).glob('Subject*.edf'))
    }:
        raise ValueError('Source EDF changed during training.')
    payload = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        rng_state=torch.get_rng_state(),
        cursor=cursor,
        completed_steps=args.steps,
        initial_checkpoint_sha256=initial_hash,
        protocol=protocol,
        training_losses=losses,
        evaluation=evaluation,
    )
    save_checkpoint(args.output_path, payload)
    print(
        json.dumps(
            dict(
                artifact=args.output_path,
                completed_steps=args.steps,
                native_states=len(backend.state_dict()),
                train_windows=len(train_rows),
                test_windows=len(test_rows),
                training_losses=losses,
                evaluation=evaluation,
            )
        )
    )


if __name__ == '__main__':
    main()
