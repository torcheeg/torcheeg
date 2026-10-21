"""
Continue complete NeuroRVQ symmetric masked-token training on EEGMat
=====================================================================

Supply trusted author source at27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c,
and both complete official EEG-v1 foundation and tokenizer checkpoints from
ntinosbarmpas/NeuroRVQ revision d944b87f44ae0ba2923b2f10d0518f23f6803b76.
They are separately licensed CC-BY-NC-4.0 and are never downloaded or bundled.
Cite Barmpas et al., NeuroRVQ, and Zyma et al. (2019), EEGMat,
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Retain all330foundation and505tokenizer native states, including32 complete
8192-class prediction heads and the teacher decoder. The teacher is frozen;
every teacher state, including usage statistics, remains unchanged. Each input
uses a random half of its patches (floor, without replacement) and its exact
complement. CE means over all32heads/selected patches are computed separately
for each partition, then SUMMED. The released source has no stage-two training
loop or reduction specification; this is an explicit TorchEEG teaching objective
using original networks, logits and targets, not a source-trainer reproduction.

Whole EDFs are resampled500->200Hz, FIR.5--45Hz/notch50 filtered; retain all19
source electrodes/reference. Each4s window/channel uses population-zscore with
std floor1e-6. Final sorted subject is held out; labels never enter training.
Fixed CPU AdamW1e-5/wd1e-5, sequential wrapping batches. --steps is TOTAL including
resume. Save full model/teacher/optimizer, CPU dropout/mask RNG, cursor, original
hashes and protocol. Validation uses first4heldout windows with an independent
fixed mask generator and no parameter selection. This is not a benchmark score.
Use a new output file, and a new cache if data or preprocessing changes.

Run::

    python examples/examples_eegmat_neurorvq_mtp_training.py --root-path ./eegmat \
        --backend-root ./NeuroRVQ --foundation-checkpoint ./NeuroRVQ_EEG_foundation_model_v1.pt \
        --tokenizer-checkpoint ./NeuroRVQ_EEG_tokenizer_v1.pt --steps 2 --output-path ./mtp2.pt

    python examples/examples_eegmat_neurorvq_mtp_training.py --root-path ./eegmat \
        --backend-root ./NeuroRVQ --resume ./mtp2.pt --steps 3 --output-path ./mtp3.pt
"""

import argparse
import hashlib
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import mne
import numpy as np
import torch
from torch import nn
import yaml

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import NeuroRVQ, NeuroRVQTokenizer, NeuroRVQMaskedTokenPretrainer

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


def build_neurorvq_backend(root, channel_names):
    """Construct a complete trusted author backend at the pretraining revision."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned NeuroRVQ pretraining revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    sys.path.insert(0, str(root))
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM
    from inference.modules.NeuroRVQ_EEG_FM_inference_modules import ch_names_global

    vocab = [name.decode() for name in ch_names_global]
    config = dict(
        n_patches=256,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        num_classes=0,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1e-5,
        n_global_electrodes=len(vocab),
        vocab_size=8192,
        use_as_encoder=True,
        use_for_pretraining=True,
    )
    backend = NeuroRVQFM(**config, qk_norm=partial(nn.LayerNorm, eps=1e-6))
    # Parameter-free feature view retains all pretrained layers and 32 heads.
    backend.norm = nn.Identity()
    for index in range(1, 5):
        setattr(backend, 'fc_norm_' + str(index), nn.Identity())
        setattr(backend, 'head_' + str(index), nn.Identity())
    config['channel_vocab'] = vocab
    return backend, config


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


def make_mask(eeg, generator):
    """Select floor(half) patches per example, without replacement, on CPU."""
    batch, channels, samples = eeg.shape
    count = channels * (samples // 200)
    if samples % 200 or count < 2 or eeg.device.type != 'cpu':
        raise ValueError(
            'Require CPU complete patches with at least two patches per example.'
        )
    order = torch.rand(batch, count, generator=generator).argsort(dim=1)
    mask = torch.zeros(batch, count, dtype=torch.bool)
    mask.scatter_(1, order[:, : count // 2], True)
    return mask.reshape(batch, channels, samples // 200)


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
        description='Continue full NeuroRVQ symmetric masked-token training on real EEGMat, CPU.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--foundation-checkpoint')
    parser.add_argument('--tokenizer-checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--cache-path', default='./eegmat-neurorvq-tokenizer-200hz-4s')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--validation-windows', type=int, default=4)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    if (
        bool(args.foundation_checkpoint) != bool(args.tokenizer_checkpoint)
        or bool(args.foundation_checkpoint) == bool(args.resume)
        or min(args.steps, args.batch_size, args.validation_windows) < 1
    ):
        parser.error(
            'Supply both full initial checkpoints OR resume, and positive counts.'
        )
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
    teacher_backend, vocab = build_neurorvq_tokenizer(args.backend_root)
    backbone_backend, backbone_config = build_neurorvq_backend(
        args.backend_root, MENTAL_ARITHMETIC_CHANNEL_LIST
    )
    if backbone_config['channel_vocab'] != vocab:
        raise ValueError('Teacher and backbone original vocabularies differ.')
    config = dict(
        channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST,
        channel_vocab=vocab,
        sampling_rate=200,
        track_usage=False,
    )
    physical = {name: value for name, value in config.items() if name != 'track_usage'}
    if args.resume:
        backbone = NeuroRVQ(backbone_backend, **physical)
        tokenizer = NeuroRVQTokenizer(teacher_backend, **config)
        initial_hash = None
    else:
        backbone = NeuroRVQ.from_pretrained(
            args.foundation_checkpoint, backbone_backend, **physical
        )
        tokenizer = NeuroRVQTokenizer.from_pretrained(
            args.tokenizer_checkpoint, teacher_backend, **config
        )
        initial_hash = dict(
            foundation=sha256(args.foundation_checkpoint),
            tokenizer=sha256(args.tokenizer_checkpoint),
        )
    model = NeuroRVQMaskedTokenPretrainer(backbone, tokenizer)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-5, weight_decay=1e-5)
    mask_generator = torch.Generator().manual_seed(17)
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
    protocol['objective'] = (
        'mean all32heads/selected patches independently per partition; SUM the two means'
    )
    protocol['mask'] = (
        'floor(half) patches per example, without replacement, CPU generator17'
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
        initial_hash = payload['initial_checkpoints_sha256']
        mask_generator.set_state(payload['mask_generator_state'])
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
        result = model(eeg, make_mask(eeg, mask_generator))
        if not torch.isfinite(result['loss']):
            raise ValueError('Nonfinite training loss.')
        result['loss'].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
            ):
                raise ValueError('Missing or nonfinite gradient: ' + name)
        optimizer.step()
        losses.append(result['loss'].item())
    model.eval()
    evaluation = []
    validation_generator = torch.Generator().manual_seed(1729)
    with torch.no_grad():
        for index in test_rows[: args.validation_windows]:
            eeg = dataset[index][0].unsqueeze(0)
            result = model(eeg, make_mask(eeg, validation_generator))
            evaluation.append(
                {
                    key: result[key].item()
                    for key in [
                        'loss',
                        'masked_loss',
                        'symmetric_loss',
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
        initial_checkpoints_sha256=initial_hash,
        mask_generator_state=mask_generator.get_state(),
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
                teacher_native_states=len(teacher_backend.state_dict()),
                foundation_native_states=len(backbone_backend.state_dict()),
                full_wrapper_states=len(model.state_dict()),
                train_windows=len(train_rows),
                test_windows=len(test_rows),
                training_losses=losses,
                evaluation=evaluation,
            )
        )
    )


if __name__ == '__main__':
    main()
