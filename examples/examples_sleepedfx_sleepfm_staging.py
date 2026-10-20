"""
Adapt complete external SleepFM staging on Sleep-EDF
====================================================

Use the separately obtained CC-BY-NC4.0 SleepFM-Clinical checkout at
2bcbae04c3592f61352addb7ac3d4193f0a3ca25. This explicit trusted local source
builds both full native models. Strictly load all 155 upstream and 37 staging
state tensors; source code and checkpoints are not bundled with TorchEEG.
Install the author's loguru dependency in your isolated environment.

Use the two physical bipolar EEG channels, preserve their source reference,
filter and resample whole records0.5--30Hz/to128Hz, then extract scored30-second epochs
with SleepEDFxDataset. Convert volts to microvolts, then use per-epoch/channel
population z-scores with std floor1e-6microvolts. Each30-second epoch becomes
six native five-second tokens. Keep it as an independent short sequence:
there is no stitching across annotation gaps or different records.
Average its six five-class logits to score the original30-second label.
This explicit short-context transfer protocol is not the published full-night
multimodal PSG benchmark or clinical validation.

Hold out the last complete subject. Select a deterministic uniform subset
of each partition without using labels, and adapt only the original final
five-class linear layer on training subjects. All other original states
remain frozen. Save complete encoder/head states and ordered metadata,
selected sample IDs, source hashes and held-out predictions, then strictly
restore and reproduce the saved predictions. No test-based tuning.

Cite SleepFM, Nature Medicine, doi:10.1038/s41591-025-04133-4, and Kemp et al.,
Sleep-EDF Expanded, https://physionet.org/content/sleep-edfx/1.0.0/.
Use those separately obtained data under their PhysioNet attribution terms.

Run::

    python examples/examples_sleepedfx_sleepfm_staging.py --root-path ./sleep-edfx \\
        --backend-root ./sleepfm-clinical --epochs 1 --max-windows 64"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from torcheeg import transforms
from torcheeg.datasets import SleepEDFxDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import SleepFM, SleepFMStaging


SOURCE_REVISION = '2bcbae04c3592f61352addb7ac3d4193f0a3ca25'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
LABELS = [
    'Sleep stage W',
    'Sleep stage N1',
    'Sleep stage N2',
    'Sleep stage N3',
    'Sleep stage R',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_backends(root):
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete SleepFM author checkout.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    module_name = (
        'sleepfm_author_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    )
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, root / 'sleepfm/models/models.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    base_config = json.loads(
        (root / 'sleepfm/checkpoints/model_base/config.json').read_text()
    )
    params = {
        k: base_config[k]
        for k in [
            'in_channels',
            'patch_size',
            'embed_dim',
            'num_heads',
            'num_layers',
            'pooling_head',
            'dropout',
        ]
    }
    stage_config = json.loads(
        (root / 'sleepfm/checkpoints/model_sleep_staging/config.json').read_text()
    )['model_params']
    base = SleepFM.from_pretrained(
        root / 'sleepfm/checkpoints/model_base/best.pt',
        module.SetTransformer(**params),
        CHANNELS,
    )
    stage = SleepFMStaging.from_pretrained(
        root / 'sleepfm/checkpoints/model_sleep_staging/best.pth',
        module.SleepEventLSTMClassifier(**stage_config),
        ['BAS'],
    )
    return base, stage


def prepare_epoch(eeg):
    values = np.asarray(eeg, dtype=np.float64) * 1e6
    if values.shape != (2, 3840) or not np.isfinite(values).all():
        raise ValueError('Require two finite bipolar channels and3840samples at128Hz.')
    return (
        (values - values.mean(-1, keepdims=True))
        / values.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)


def label_index(label):
    return LABELS.index(label)


def select_windows(dataset, maximum):
    if len(dataset) == 0:
        raise ValueError('Require nonempty subject partitions.')
    indices = np.linspace(
        0, len(dataset) - 1, min(maximum, len(dataset)), dtype=int
    ).tolist()
    return Subset(dataset, indices), dataset.info.iloc[indices].clip_id.astype(
        str
    ).tolist()


def predict(base, stage, eeg):
    with torch.no_grad():
        features = base(eeg, return_tokens=True).unsqueeze(1)
    logits, padding = stage(features)
    if padding.any():
        raise ValueError('Independent30-second epochs do not contain padded time.')
    return logits.mean(1)


def main():
    parser = argparse.ArgumentParser(
        description='Adapt full native SleepFM final staging layer with subject holdout.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./sleepedf-sleepfm-128hz-30s')
    parser.add_argument('--output-path', default='./sleepedf-sleepfm-staging.pt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-windows', type=int, default=64)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_windows) < 1:
        parser.error('epochs, batch-size and max-windows must be positive.')
    torch.manual_seed(42)
    base, stage = build_backends(args.backend_root)
    base.requires_grad_(False).eval()
    stage.requires_grad_(False).eval()
    stage.backend.fc.requires_grad_(True)
    dataset = SleepEDFxDataset(
        root_path=args.root_path,
        studies=['cassette'],
        channels=CHANNELS,
        missing_channels='raise',
        sfreq=128,
        l_freq=0.5,
        h_freq=30,
        io_path=args.cache_path,
        io_mode='pickle',
        num_worker=0,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_epoch), transforms.ToTensor()]
        ),
        label_transform=transforms.Compose(
            [transforms.Select('label'), transforms.Lambda(label_index)]
        ),
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('Require at least two subjects for a complete held-out split.')
    training, testing = next(
        LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    train, train_ids = select_windows(training, args.max_windows)
    test, test_ids = select_windows(testing, args.max_windows)
    source = Path(args.root_path)
    hashes = {
        str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.edf'))
    }
    original_base = {k: v.clone() for k, v in base.backend.state_dict().items()}
    original_stage = {k: v.clone() for k, v in stage.backend.state_dict().items()}
    optimizer = torch.optim.Adam(stage.backend.fc.parameters(), lr=0.0001)
    updates = 0
    for _ in range(args.epochs):
        for eeg, target in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(predict(base, stage, eeg), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss must remain finite.')
            loss.backward()
            optimizer.step()
            updates += 1

    def evaluate(encoder, head):
        logits, targets = [], []
        with torch.no_grad():
            for eeg, target in DataLoader(test, batch_size=args.batch_size):
                logits.append(predict(encoder, head, eeg))
                targets.append(target)
        return torch.cat(logits), torch.cat(targets)

    logits, targets = evaluate(base, stage)
    metrics = dict(
        loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item(),
    )
    for k, v in original_base.items():
        torch.testing.assert_close(base.backend.state_dict()[k], v, rtol=0, atol=0)
    for k, v in original_stage.items():
        if not k.startswith('fc.'):
            torch.testing.assert_close(stage.backend.state_dict()[k], v, rtol=0, atol=0)
    metadata = dict(
        variant='sleep-staging',
        backend_revision=SOURCE_REVISION,
        weights_license='CC-BY-NC-4.0',
        source_sha256=hashes,
        channels=CHANNELS,
        modality_names=['BAS'],
        sampling_rate=128,
        reference='original bipolar EDF reference preserved',
        window_samples=3840,
        preprocessing='reader whole-record0.5--30HzFIR and whole-record resample128Hz, volts->uV, per-epoch/channel populationzscore stdfloor1e-6uV',
        context='six5-secondtokens within each independent30-secondepoch; mean five-class logits',
        label_mapping={name: index for index, name in enumerate(LABELS)},
        train_subjects=subjects[:-1],
        test_subject=subjects[-1],
        train_ids=train_ids,
        test_ids=test_ids,
        epochs=args.epochs,
        updates=updates,
        batch_size=args.batch_size,
        optimizer={'name': 'Adam', 'lr': 0.0001},
        seed=42,
        metrics=metrics,
        checkpoint_sha256={
            name: sha256(Path(args.backend_root) / 'sleepfm/checkpoints' / name / file)
            for name, file in [
                ('model_base', 'best.pt'),
                ('model_sleep_staging', 'best.pth'),
            ]
        },
        only_original_final_linear_layer_adapted=True,
        benchmark_claim=False,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            base_state=base.state_dict(),
            staging_state=stage.state_dict(),
            metadata_json=json.dumps(metadata),
            test_logits=logits,
            test_targets=targets,
        ),
        output,
    )
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored_base, restored_stage = build_backends(args.backend_root)
    restored_base.load_state_dict(payload['base_state'], strict=True)
    restored_stage.load_state_dict(payload['staging_state'], strict=True)
    restored_base.eval()
    restored_stage.eval()
    restored_logits, restored_targets = evaluate(restored_base, restored_stage)
    torch.testing.assert_close(restored_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(restored_targets, targets, rtol=0, atol=0)
    if hashes != {
        str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.edf'))
    }:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, strict_restore=True, artifact=str(output))))


if __name__ == '__main__':
    main()
