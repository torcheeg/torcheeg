"""
Train the complete SleepFM diagnosis model on an explicit survival cohort
=========================================================================

Supply native five-second embeddings, measured follow-up/event tables and
already encoded demographics through SleepFMSurvivalDataset. The manifest
declares patient-separated train/validation/test membership and every input
identity. This workflow uses the complete official diagnosis model and the
grouped Breslow objective, not classification labels or a sigmoid.

The CPU teaching loop uses distinct patients per minibatch, merges a final
singleton into the previous batch, and steps the native AdamW/plateau schedule
at completed epochs. Validation reports the mean of its fixed batch risk-set
losses, not a full-cohort likelihood. Test outcomes are not used for selection.
The total --steps includes resumed steps; exports restore the complete model,
optimizer, scheduler, patient permutation/cursor and both Torch random states.

Author source and weights remain separately supplied CC-BY-NC materials.
Import does not load or download them. Synthetic schema fixtures and the
author's one-subject synthetic demo are not real clinical training evidence.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import torch

from torcheeg.datasets import SleepFMSurvivalDataset, collate_sleepfm_survival
from torcheeg.models import SleepFMDiagnosis, SleepFMSurvivalObjective


SOURCE_REVISION = '2bcbae04c3592f61352addb7ac3d4193f0a3ca25'
SOURCE_FILES = {
    'sleepfm/models/models.py': 'b4f66873f863c0205bba14d02ba27d54de623b038c674144f6ff04b742f81e9c',
    'sleepfm/checkpoints/model_diagnosis/config.json': 'dc7faac48d73254815167fd252dae31daaaf6843bfd8139a139261cad751a8cc',
    'sleepfm/checkpoints/model_diagnosis/best.pth': '78e4744c5b23e98dbfd779dd523e2f666cdb646db40c6cc56d21060aae3706df',
    'sleepfm/checkpoints/model_base/best.pt': 'ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799',
}


def build_diagnosis(backend_root, schema):
    """Explicitly construct the complete trusted source model and load all35states."""
    root = Path(backend_root).resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Require the documented pinned complete SleepFM release.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    for name, expected in SOURCE_FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                'Trusted source/config/official checkpoint differs: ' + name
            )
    if (
        schema['upstream_source_revision'] != SOURCE_REVISION
        or schema['upstream_checkpoint_sha256']
        != SOURCE_FILES['sleepfm/checkpoints/model_base/best.pt']
    ):
        raise ValueError(
            'Cohort embeddings must declare the matching original upstream source/checkpoint.'
        )
    source = root / 'sleepfm/models/models.py'
    name = (
        '_torcheeg_sleepfm_survival_'
        + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    if Path(module.__file__).resolve() != source:
        raise ValueError('The explicit source module belongs to a different checkout.')
    config = json.loads(
        (root / 'sleepfm/checkpoints/model_diagnosis/config.json').read_text()
    )
    if config['model'] != 'DiagnosisFinetuneFullLSTMCOXPHWithDemo':
        raise ValueError('Require the complete released diagnosis architecture.')
    backend = module.DiagnosisFinetuneFullLSTMCOXPHWithDemo(**config['model_params'])
    model = SleepFMDiagnosis.from_pretrained(
        root / 'sleepfm/checkpoints/model_diagnosis/best.pth',
        backend,
        schema['modality_names'],
        schema['demographic_names'],
    )
    if len(backend.state_dict()) != 35:
        raise ValueError('Do not replace or discard any original diagnosis state.')
    return model, dict(source_revision=revision, files_sha256=SOURCE_FILES.copy())


def batch_ranges(count, batch_size):
    """Use every patient once, merging a last singleton without duplication."""
    if (
        type(count) is not int
        or type(batch_size) is not int
        or count < 2
        or batch_size < 2
    ):
        raise ValueError('Require at least two distinct patients and batch_size>=2.')
    ranges = []
    start = 0
    while start < count:
        end = min(start + batch_size, count)
        if count - end == 1:
            end = count
        ranges.append((start, end))
        start = end
    return ranges


def _forward(objective, batch):
    return objective(
        batch['embeddings'],
        batch['demographics'],
        batch['event_times'],
        batch['is_event'],
        batch['padding_mask'],
    )


def evaluate(objective, dataset, batch_size):
    """Read-only held-out fixed risk sets; restore training mode and Torch RNG."""
    rng, training = torch.get_rng_state(), objective.training
    objective.eval()
    losses, risks, studies, subjects = [], [], [], []
    try:
        with torch.no_grad():
            for start, end in batch_ranges(len(dataset), batch_size):
                batch = collate_sleepfm_survival(
                    [dataset[i] for i in range(start, end)]
                )
                result = _forward(objective, batch)
                losses.append(result['loss'].item())
                risks.append(result['log_risks'].cpu())
                studies.extend(batch['study_id'])
                subjects.extend(batch['subject_id'])
    finally:
        objective.train(training)
        torch.set_rng_state(rng)
    return dict(
        loss=sum(losses) / len(losses),
        batch_losses=losses,
        log_risks=torch.cat(risks),
        study_id=studies,
        subject_id=subjects,
    )


def _validate_cursor(cursor, completed_steps, count, batch_size):
    if not isinstance(cursor, dict) or set(cursor) != {'epoch', 'offset', 'order'}:
        raise ValueError('Require a complete patient-permutation cursor.')
    epoch, offset, order = cursor['epoch'], cursor['offset'], cursor['order']
    ranges = batch_ranges(count, batch_size)
    starts = [start for start, _ in ranges]
    if (
        type(epoch) is not int
        or epoch < 0
        or type(offset) is not int
        or offset not in starts
        or not isinstance(order, torch.Tensor)
        or order.dtype != torch.int64
        or order.device.type != 'cpu'
        or order.ndim != 1
        or order.numel() not in (0, count)
    ):
        raise ValueError('Patient cursor, offset or permutation type is invalid.')
    if order.numel() == 0:
        if offset != 0:
            raise ValueError('An empty permutation must start a new epoch.')
    elif not torch.equal(
        order.sort().values, torch.arange(count, dtype=torch.int64, device='cpu')
    ):
        raise ValueError(
            'Every patient must appear exactly once in an epoch permutation.'
        )
    if completed_steps != epoch * len(ranges) + starts.index(offset):
        raise ValueError('Completed steps disagree with the epoch/permutation cursor.')


def save_checkpoint(path, payload):
    """Publish complete bytes atomically without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix='.tmp', delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--event-times-path', required=True)
    parser.add_argument('--is-event-path', required=True)
    parser.add_argument('--demographics-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument(
        '--steps',
        type=int,
        default=3,
        help='Total completed optimizer steps, including resume.',
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume')
    args = parser.parse_args(argv)
    output = Path(args.output_path)
    if output.exists():
        parser.error('output-path exists; select a new file.')
    if args.steps < 1 or args.batch_size < 2 or not 0 <= args.seed < 2**32:
        parser.error('Require steps>=1, batch-size>=2 and seed in [0,2**32).')
    paths = dict(
        manifest_path=args.manifest_path,
        event_times_path=args.event_times_path,
        is_event_path=args.is_event_path,
        demographics_path=args.demographics_path,
    )
    train = SleepFMSurvivalDataset(**paths, split='train')
    validation = SleepFMSurvivalDataset(**paths, split='validation')
    ranges = batch_ranges(len(train), args.batch_size)
    batch_ranges(len(validation), args.batch_size)
    if train.provenance != validation.provenance:
        raise ValueError('Train/validation inputs changed while reading the cohort.')
    train.validate_files()
    validation.validate_files()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    diagnosis, source = build_diagnosis(args.backend_root, train.schema)
    objective = (
        SleepFMSurvivalObjective(diagnosis, list(train.endpoint_names)).cpu().train()
    )
    optimizer = torch.optim.AdamW(objective.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )
    protocol = dict(
        version=1,
        cohort=train.provenance,
        backend=source,
        seed=args.seed,
        batch_size=args.batch_size,
        optimizer='AdamW_lr0.001_weight_decay0.01',
        scheduler='ReduceLROnPlateau_factor0.1_patience10_completed_epochs',
        validation='fixed_manifest_batches_mean_loss_merge_final_singleton_no_test_selection',
        training_study_ids=train.info['study_id'].tolist(),
        validation_study_ids=validation.info['study_id'].tolist(),
        example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256={
            cls.__name__: hashlib.sha256(
                Path(sys.modules[cls.__module__].__file__).read_bytes()
            ).hexdigest()
            for cls in (
                SleepFMSurvivalDataset,
                SleepFMDiagnosis,
                SleepFMSurvivalObjective,
            )
        },
        runtime=dict(
            torch=str(torch.__version__), machine=platform.machine(), cpu_threads=2
        ),
    )
    protocol_json = json.dumps(protocol, sort_keys=True)
    order_rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    completed_steps = 0
    cursor = dict(
        epoch=0, offset=0, order=torch.empty(0, dtype=torch.int64, device='cpu')
    )
    history, training_batches = [], []
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location='cpu')
        if not isinstance(saved, dict) or saved.get('protocol_json') != protocol_json:
            raise ValueError(
                'Checkpoint cohort/schema/input/source/teaching protocol differs.'
            )
        completed_steps = saved.get('completed_steps')
        if type(completed_steps) is not int or not 0 <= completed_steps < args.steps:
            raise ValueError(
                'Resume steps must increase the total completed optimizer steps.'
            )
        cursor = saved['cursor']
        _validate_cursor(cursor, completed_steps, len(train), args.batch_size)
        history, training_batches = saved['training_losses'], saved['training_batches']
        if (
            not isinstance(history, list)
            or len(history) != completed_steps
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in history
            )
            or not isinstance(training_batches, list)
            or len(training_batches) != completed_steps
        ):
            raise ValueError('Training history/cursor is incomplete or nonfinite.')
        objective.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng']['torch'])
        order_rng.set_state(saved['rng']['data_order'])
    last_evaluation, last_evaluation_step = None, -1
    while completed_steps < args.steps:
        if cursor['order'].numel() == 0:
            cursor['order'] = torch.randperm(
                len(train), generator=order_rng, device='cpu'
            )
        start = cursor['offset']
        end = next(end for begin, end in ranges if begin == start)
        indices = cursor['order'][start:end].tolist()
        batch = collate_sleepfm_survival([train[i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        result = _forward(objective, batch)
        result['loss'].backward()
        if any(
            not torch.isfinite(p.grad).all()
            for p in objective.parameters()
            if p.grad is not None
        ):
            raise ValueError(
                'Nonfinite model gradient; no optimizer update is applied.'
            )
        optimizer.step()
        completed_steps += 1
        history.append(result['loss'].item())
        training_batches.append(batch['study_id'])
        cursor['offset'] = end
        if end == len(train):
            cursor = dict(
                epoch=cursor['epoch'] + 1,
                offset=0,
                order=torch.empty(0, dtype=torch.int64, device='cpu'),
            )
            last_evaluation = evaluate(objective, validation, args.batch_size)
            last_evaluation_step = completed_steps
            scheduler.step(last_evaluation['loss'])
    _validate_cursor(cursor, completed_steps, len(train), args.batch_size)
    if last_evaluation_step != completed_steps:
        last_evaluation = evaluate(objective, validation, args.batch_size)
    payload = dict(
        protocol_json=protocol_json,
        state_dict=objective.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        completed_steps=completed_steps,
        cursor=cursor,
        training_losses=history,
        training_batches=training_batches,
        evaluation=last_evaluation,
        rng=dict(torch=torch.get_rng_state(), data_order=order_rng.get_state()),
    )
    save_checkpoint(output, payload)
    print(
        json.dumps(
            dict(
                data_kind=train.schema['data_kind'],
                train_patients=len(train),
                validation_patients=len(validation),
                completed_steps=completed_steps,
                full_states=len(objective.state_dict()),
                validation_batch_risk_loss=last_evaluation['loss'],
            )
        )
    )
    return payload


if __name__ == '__main__':
    main()
