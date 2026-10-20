"""
Probe pretrained LUNA-Huge on EEGMat
====================================

Load the complete official LUNA-huge encoder/reconstruction checkpoint,
freeze the backbone, and train the native aggregation-query classification
head on background=0 versus mental arithmetic=1. Cite LUNA/BioFoundation
and Zyma et al. (2019), EEGMat, https://physionet.org/content/eegmat/1.0.0/.
The EEGMat recordings use ODC Attribution v1.0. The model source is Apache2;
its official weights separately use CC-BY-ND4.0. Keep trained weight artifacts
local unless their original license permits the intended sharing. This
example publishes no weights and downloads no data or models.

Resample complete recordings from 500 to256Hz using MentalArithmeticDataset,
then filter whole recordings at0.1--75Hz and notch50Hz with MNE FIR filters.
This explicit after-resampling teaching protocol differs from source pipelines
that filter before resampling. Split nonoverlapping five-second windows
(1280samples,32patches40). Normalize each source channel/window to zero mean
and population standard deviation, clamped below at1e-8microvolts. No learned
population statistic is fit. Preserve all19source channels and their source
reference. Coordinates come from MNE standard1005 in head-frame metres,
matched to that exact channel order; they are supplied explicitly to LUNA.

Hold out the last sorted complete subject and train on all others for a fixed
epoch count. Test labels never select epochs or settings. A short teaching run
verifies model/data functionality, not replication of a published benchmark.
Supply the full original LUNA_huge.safetensors from author revision
999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f at https://huggingface.co/PulpBio/LUNA.
Verify the SHA256 listed in the guide. There is no small/random fallback.

Save the full tensor state (including unused reconstruction components),
constructor/preprocessing JSON, source/weight hashes, coordinate tensor,
subject split and held-out predictions. Strictly restore the complete model
and reproduce predictions before reporting success. Original/modified weight
license terms still apply to local exports. Use a new cache path when source
recordings or offline processing changes.

Run::

    python examples/examples_eegmat_luna_huge.py --root-path ./eegmat \
        --checkpoint ./LUNA_huge.safetensors --epochs 1
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import LUNA, LUNAClassifier


VARIANT = 'huge'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=256,
        l_freq=.1, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=256, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 1280) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 1280 samples at256Hz.')
    eeg = np.asarray(eeg, dtype=np.float64)
    mean = eeg.mean(axis=-1, keepdims=True)
    scale = np.maximum(eeg.std(axis=-1, keepdims=True), 1e-8)
    return ((eeg - mean) / scale).astype(np.float32)


def electrode_positions():
    info = mne.create_info(MENTAL_ARITHMETIC_CHANNEL_LIST, sfreq=256, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False)
    locations = info.get_montage().get_positions()['ch_pos']
    return torch.tensor(np.stack([locations[name] for name in MENTAL_ARITHMETIC_CHANNEL_LIST]), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete pretrained LUNA-huge EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-luna-huge-256hz-5s')
    parser.add_argument('--output-path', default='./eegmat-luna-huge.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backbone = LUNA.from_pretrained(args.checkpoint, variant=VARIANT)
    positions = electrode_positions()
    model = LUNAClassifier(backbone, num_classes=2, freeze_backbone=True, positions=positions)
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=1280,
        num_channel=19, sampling_rate=256, io_path=args.cache_path, io_mode='pickle',
        num_worker=0, before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(prepare_window), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two complete subject pairs are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
        dataset, subject=subjects[-1]))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.001)
    updates = 0
    for _ in range(args.epochs):
        model.train()
        for eeg, target in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(eeg), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in DataLoader(test, batch_size=args.batch_size):
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    metrics = dict(loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item())
    config = dict(patch_size=backbone.patch_size, hid_channels=backbone.hid_channels,
        num_queries=backbone.num_queries, depth=backbone.depth, num_heads=backbone.num_heads,
        mlp_ratio=backbone.mlp_ratio, drop_path=backbone.drop_path)
    metadata = dict(variant=VARIANT, checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        coordinate_frame='MNE Info.set_montage standard_1005 head-frame metres',
        sampling_rate=256, window_samples=1280, input_unit='microvolts before normalization',
        normalization='per-channel/window population z-score; std clamped at1e-8microvolts',
        filters='MNE FIR0.1--75Hz and50Hznotch on complete recordings AFTER resampling to256Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_backbone=True,
        weights_license='CC-BY-ND-4.0; see original model card before sharing local exports')
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), positions=positions, test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = LUNAClassifier(LUNA(**json.loads(payload['backbone_config_json'])), 2, freeze_backbone=True, positions=payload['positions']).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
