"""
Continue complete Uni-DMFM self-supervised training on EEGMat
=============================================================

Explicitly supply trusted ichenws/Uni-DMFM source at
522f9fc09b6d5f3d1051e6941d1bff548d8f10de and the complete official
pretrained_weights.pth (required also for resume to establish all144profiling
buffers). No author source or weights are bundled; their pinned release has
no stated license. Obtain them under applicable original terms. The complete
12-layer384-wide network and all386states/three heads remain present.

Call original source augmentations, span masking and NTXentLoss explicitly.
Mask only the first augmented view; reconstruct the CLEAN original waveform
and log(abs(STFT)+1e-6) targets with masked MSE, plus native NT-Xent(temp.2).
Span lengths5--10 clip and overlap: mask_ratio.6 is approximate, not exact.
The original loss_mae names actually denote squared error. Preserve them as MSE.

Real EEGMat whole-record500->200Hz, FIR.1--75/notch50, source reference, all19
physical electrodes,20s4000sample windows. Convert float32 BEFORE per-channel
unbiased standardization /(std+1e-6), clamp[-10,10], as native pretraining.
Final sorted subject is held out. Labels never enter optimization. Batch2
provides actual negatives; batches sequentially wrap without duplicate rows.
This small CPUfloat32 teaching transfer is not TUEG or a published benchmark.

Use native AdamW1e-4/wd.05 and gradientclip1. Teaching cosine schedule is fixed
at1000updates/100warmup/min-start1e-6, independent of --steps (TOTAL including
resume). Save every model/profiling/window/policy state, AdamW/scheduler, global
Torch/Python/NumPy RNG, cursor, protocol and raw/cache hashes. Validation uses
first4held-out windows with separate fixed RNG; restore the training RNG after
validation. No CUDA AMP/scaler or distributed execution is selected here.

Cite Uni-DMFM, https://doi.org/10.1016/j.patcog.2026.114221, and Zyma et al.
(2019), https://physionet.org/content/eegmat/1.0.0/ (ODC Attributionv1.0).
Install timm for the original backend; lmdb is already a TorchEEG dependency.

Run::

    python examples/examples_eegmat_unidmfm_pretraining.py --root-path ./eegmat \
        --backend-root ./Uni-DMFM --checkpoint ./pretrained_weights.pth \
        --steps 2 --output-path ./pretrain2.pt
    python examples/examples_eegmat_unidmfm_pretraining.py --root-path ./eegmat \
        --backend-root ./Uni-DMFM --checkpoint ./pretrained_weights.pth \
        --resume ./pretrain2.pt --steps 3 --output-path ./pretrain3.pt
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import mne
import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import UniDMFM, UniDMFMPretrainer

SOURCE_REVISION = '522f9fc09b6d5f3d1051e6941d1bff548d8f10de'
CHECKPOINT_SHA256 = '92fc5f59cd3ffc1c25898f1a573db8d8194935a163bbcb9c68d45f86ba81d5fe'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_unidmfm_backend(root, checkpoint_path, channel_names):
    """Explicitly construct the full trusted source and released profiling state."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete Uni-DMFM author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError(
            'Use the documented complete original checkpoint for backend construction.'
        )
    name = (
        '_torcheeg_external_unidmfm_'
        + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'models/model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    backend = sys.modules[name].UniDMFM()
    state = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True, mmap=True
    )['model']
    expected = backend.state_dict()
    extra = set(state) - set(expected)
    if len(expected) != 242 or not set(expected) <= set(state) or len(extra) != 144:
        raise ValueError(
            'Require all original neural tensors and the released profiling buffers.'
        )
    for key in expected:
        if (
            state[key].shape != expected[key].shape
            or state[key].dtype != expected[key].dtype
        ):
            raise ValueError('Native neural shape/type mismatch: ' + key)
    for key in sorted(extra):
        path, _, buffer = key.rpartition('.')
        value = state[key]
        if (
            buffer not in ('total_ops', 'total_params')
            or value.shape != (1,)
            or value.dtype != torch.float64
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError('Invalid original nonneural profiling buffer: ' + key)
        target = backend.get_submodule(path) if path else backend
        target.register_buffer(buffer, torch.zeros_like(value))
    return backend, dict(
        num_layers=12,
        nhead=12,
        d_model=384,
        patch_size=200,
        stft_params=dict(backend.stft_params),
        drop_path_rate=0.1,
        profiling_buffers=144,
    )


def load_native_pretraining(root):
    """Import trusted pinned helpers explicitly, rejecting namespace collisions."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Require the documented pinned original pretraining source.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    for name, relative in [
        ('models.model', 'models/model.py'),
        ('util.utils', 'util/utils.py'),
    ]:
        existing = sys.modules.get(name)
        if existing is not None:
            filename = getattr(existing, '__file__', None)
            if filename is None or Path(filename).resolve() != root / relative:
                raise ValueError(
                    'Another author source namespace is imported; use a fresh process.'
                )
    sys.path.insert(0, str(root))
    name = (
        '_torcheeg_external_unidmfm_pretraining_'
        + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'pretrain.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    return module


def filter_recording(eeg):
    values = mne.filter.filter_data(
        np.asarray(eeg, dtype=np.float64),
        sfreq=200,
        l_freq=0.1,
        h_freq=75.0,
        verbose=False,
    )
    return mne.filter.notch_filter(values, Fs=200, freqs=[50.0], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 4000) or not np.isfinite(eeg).all():
        raise ValueError('Require finite source19channels/20seconds at200Hz.')
    patches = torch.from_numpy(np.asarray(eeg, dtype=np.float32)).reshape(19, 20, 200)
    mean = patches.mean(dim=(1, 2), keepdim=True)
    std = patches.std(dim=(1, 2), keepdim=True)
    return ((patches - mean) / (std + 1e-6)).clamp(-10, 10).reshape(19, 4000)


def capture_rng():
    state = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=dict(
            kind=state[0],
            keys=torch.from_numpy(state[1].astype(np.int64)),
            position=state[2],
            has_gauss=state[3],
            cached_gaussian=state[4],
        ),
    )


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    saved = state['numpy']
    np.random.set_state(
        (
            saved['kind'],
            saved['keys'].numpy().astype(np.uint32),
            saved['position'],
            saved['has_gauss'],
            saved['cached_gaussian'],
        )
    )


def prepare_views(ops, eeg, params):
    patches = eeg.reshape(eeg.shape[0], eeg.shape[1], -1, 200)
    first = ops.augmentations(patches, params)
    second = ops.augmentations(patches, params)
    mask = ops.generate_span_mask_vectorized(first.shape[:3], 0.6, eeg.device)
    return first.reshape_as(eeg), second.reshape_as(eeg), mask


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


def main():
    parser = argparse.ArgumentParser(
        description='Complete native Uni-DMFM self-supervised EEGMat teaching workflow, CPU.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--cache-path', default='./eegmat-unidmfm-pretrain-200hz-20s')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--validation-windows', type=int, default=4)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.validation_windows < 2:
        parser.error(
            'Require positive total steps and at least two examples for negatives.'
        )
    if Path(args.output_path).exists():
        parser.error('output-path exists; select a new file.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    files = sorted(Path(args.root_path).glob('Subject*.edf'))
    if not files:
        raise ValueError('Supply real EEGMat source EDFs.')
    hashes = {path.name: sha256(path) for path in files}
    cache = Path(args.cache_path)
    marker = cache / 'unidmfm-pretraining.json'
    preprocessing = dict(
        raw_sha256=hashes,
        channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=200,
        chunk_size=4000,
        filters='whole-record FIR.1--75/notch50 after resample200Hz',
        mne_version=str(mne.__version__),
    )
    if marker.exists():
        if json.loads(marker.read_text()) != preprocessing:
            raise ValueError('Cache input/protocol differs; use a new path.')
    elif cache.exists() and any(cache.iterdir()):
        raise ValueError('Existing cache has no verified manifest; use a new path.')
    dataset = MentalArithmeticDataset(
        root_path=args.root_path,
        chunk_size=4000,
        num_channel=19,
        sampling_rate=200,
        io_path=str(cache),
        io_mode='pickle',
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Lambda(prepare_window),
        label_transform=transforms.Select('label'),
    )
    if not marker.exists():
        marker.write_text(json.dumps(preprocessing, indent=2) + '\n')
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2 or any(
        set(dataset.info.loc[dataset.info.subject_id == s, 'label']) != {0, 1}
        for s in subjects
    ):
        raise ValueError('Require at least two complete subject pairs.')
    train_rows = dataset.info.index[dataset.info.subject_id != subjects[-1]].tolist()
    test_rows = dataset.info.index[dataset.info.subject_id == subjects[-1]].tolist()
    if args.batch_size > len(train_rows):
        raise ValueError('Batch exceeds distinct training windows.')
    backend, config = build_unidmfm_backend(
        args.backend_root, args.checkpoint, MENTAL_ARITHMETIC_CHANNEL_LIST
    )
    ops = load_native_pretraining(args.backend_root)
    backbone = (
        UniDMFM(backend, MENTAL_ARITHMETIC_CHANNEL_LIST)
        if args.resume
        else UniDMFM.from_pretrained(
            args.checkpoint, backend, MENTAL_ARITHMETIC_CHANNEL_LIST
        )
    )
    model = UniDMFMPretrainer(backbone, ops.NTXentLoss(temperature=0.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = ops.CosineLRScheduler(
        optimizer,
        t_initial=1000,
        lr_min=1e-6,
        warmup_t=100,
        warmup_lr_init=1e-6,
        cycle_limit=1,
        t_in_epochs=False,
    )
    params = SimpleNamespace(
        use_augmentations=True,
        time_shift_prob=0.5,
        max_time_shift_ratio=0.1,
        channel_dropout_ratio=0.2,
        noise_level=0.01,
    )
    protocol = dict(
        preprocessing=preprocessing,
        normalization='float32 before per-channel unbiased zscore/(std+1e-6), clamp[-10,10]',
        source_revision=SOURCE_REVISION,
        checkpoint_sha256=sha256(args.checkpoint),
        native_config=config,
        batch_size=args.batch_size,
        train_rows=train_rows,
        test_rows=test_rows,
        held_out_subject=subjects[-1],
        augmentation=vars(params),
        mask_ratio=0.6,
        objective=dict(
            time_weight=1.0,
            frequency_weight=1.0,
            contrastive_weight=1.0,
            temperature=0.2,
        ),
        optimizer=dict(name='AdamW', lr=1e-4, weight_decay=0.05, grad_clip_norm=1.0),
        scheduler=dict(
            name='CosineLRScheduler',
            t_initial=1000,
            warmup_t=100,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
        ),
        precision='CPUfloat32, no AMP/scaler',
        reference='source EDF unchanged',
        input_unit='microvolts before normalization',
        seed=42,
    )
    completed, cursor = 0, 0
    if args.resume:
        payload = torch.load(args.resume, weights_only=True, map_location='cpu')
        if payload['protocol'] != protocol:
            raise ValueError('Resume data/protocol differs.')
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        completed, cursor = payload['completed_steps'], payload['cursor']
        restore_rng(payload['rng'])
    if args.steps <= completed:
        raise ValueError('--steps must exceed the saved total.')
    losses = []
    for step in range(completed, args.steps):
        rows = [
            train_rows[(cursor + i) % len(train_rows)] for i in range(args.batch_size)
        ]
        cursor = (cursor + args.batch_size) % len(train_rows)
        eeg = torch.stack([dataset[index][0] for index in rows])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(eeg, *prepare_views(ops, eeg, params))
        if not torch.isfinite(result['loss']):
            raise ValueError('Nonfinite loss.')
        result['loss'].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
            ):
                raise ValueError('Missing or nonfinite gradient: ' + name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step_update(step)
        losses.append(result['loss'].item())
    training_rng = capture_rng()
    evaluation = []
    try:
        torch.manual_seed(1729)
        random.seed(1729)
        np.random.seed(1729)
        model.eval()
        validation = test_rows[: args.validation_windows]
        with torch.no_grad():
            for offset in range(0, len(validation), args.batch_size):
                eeg = torch.stack(
                    [
                        dataset[index][0]
                        for index in validation[offset : offset + args.batch_size]
                    ]
                )
                result = model(eeg, *prepare_views(ops, eeg, params))
                evaluation.append(
                    {
                        key: result[key].item()
                        for key in [
                            'loss',
                            'time_loss',
                            'frequency_loss',
                            'contrastive_loss',
                        ]
                    }
                )
    finally:
        restore_rng(training_rng)
    if hashes != {
        path.name: sha256(path)
        for path in sorted(Path(args.root_path).glob('Subject*.edf'))
    }:
        raise ValueError('Raw EDFs changed during the run.')
    payload = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        rng=capture_rng(),
        cursor=cursor,
        completed_steps=args.steps,
        protocol=protocol,
        training_losses=losses,
        evaluation=evaluation,
    )
    save_checkpoint(args.output_path, payload)
    print(
        json.dumps(
            dict(
                artifact=args.output_path,
                native_states=len(backend.state_dict()),
                wrapper_states=len(model.state_dict()),
                train_windows=len(train_rows),
                test_windows=len(test_rows),
                training_losses=losses,
                evaluation=evaluation,
            )
        )
    )


if __name__ == '__main__':
    main()
