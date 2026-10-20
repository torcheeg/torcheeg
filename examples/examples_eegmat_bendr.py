"""
Probe the complete pretrained BENDR on EEGMat
=============================================

Load both official BENDR encoder.pt and contextualizer.pt, freeze the complete
20-input/512-wide/eight-layer backbone, and train a linear classification
head on background=0 versus mental arithmetic=1. The public EEGMat source is
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0; Zyma et al.,
2019). Cite the dataset, PhysioNet and Kostas et al. (2021), BENDR.

The reader resamples complete 500 Hz recordings to 256 Hz and returns
microvolts. Use four-second nonoverlapping windows. Reorder all 19 scalp
channels to DN3 To1020 order, with T3/T4 interpreted as T7/T8 aliases. Apply
the native Deep1010 global EEG min/max normalization including its unused
zero rows, then retain the 19 available electrodes. Append the native unknown
max_scale auxiliary value zero as channel20. That channel is not another EEG
electrode. No source electrode is fabricated. No extra filter, artifact
removal or data-dependent population fit occurs. The model cannot infer
these preprocessing semantics from input shape.

Hold out the last sorted subject; use all other subjects for training.
Use a fixed epoch count without test-based selection. A short teaching run
checks functionality, not replication of published BENDR benchmark scores.
The example requires both complete official local weight files from
https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha. Check their SHA256
in the BENDR guide. Nothing is downloaded, and random weights are never used
as a fallback. Complete contextualized start-token features feed the head;
the convolutional encoder alone is a different representation.

Save full model state, constructor/preprocessing JSON, checkpoint/source
hashes, subject split, labels and held-out logits. Safely reload the entire
model and strictly reproduce saved predictions. Keep a fresh cache path
when source or preprocessing changes.

Run::

    python examples/examples_eegmat_bendr.py --root-path ./eegmat \
        --encoder-checkpoint ./encoder.pt --contextualizer-checkpoint ./contextualizer.pt \
        --cache-path ./cache-eegmat-bendr --output-path ./bendr-probe.pt --epochs 1
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import BENDR, BENDRClassifier


CHANNEL_ORDER = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


def prepare_window(eeg):
    if eeg.shape != (19, 1024) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 1024 samples at 256 Hz.')
    indices = [MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
    ordered = np.asarray(eeg, dtype=np.float64)[indices]
    # Native Deep1010 includes zero-valued unmapped EEG rows in global extrema.
    low, high = min(0., ordered.min()), max(0., ordered.max())
    normalized = 2 * ((ordered - low) / (high - low) - .5) if high > low else np.zeros_like(ordered)
    # Native max_scale=None fallback: unknown scale auxiliary equals zero.
    return np.concatenate((normalized, np.zeros((1, 1024))), axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete pretrained BENDR EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--encoder-checkpoint', required=True)
    parser.add_argument('--contextualizer-checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-bendr-256hz-4s')
    parser.add_argument('--output-path', default='./eegmat-bendr.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backbone = BENDR.from_pretrained(args.encoder_checkpoint, args.contextualizer_checkpoint)
    model = BENDRClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=1024,
        num_channel=19, sampling_rate=256, io_path=args.cache_path, io_mode='pickle',
        num_worker=0, online_transform=transforms.Compose([
            transforms.Lambda(prepare_window), transforms.ToTensor()]),
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
    config = dict(num_electrodes=20, hid_channels=512, mlp_channels=3076, num_heads=8,
        depth=8, encoder_dropout=0., dropout=.15, layer_drop=0.)
    metadata = dict(encoder_sha256=hashlib.sha256(Path(args.encoder_checkpoint).read_bytes()).hexdigest(),
        contextualizer_sha256=hashlib.sha256(Path(args.contextualizer_checkpoint).read_bytes()).hexdigest(),
        source_sha256=source_hashes, source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        channel_order=CHANNEL_ORDER + ['SCALE'], sampling_rate=256, window_samples=1024,
        input_unit='microvolts from MentalArithmeticDataset; normalized before the model',
        normalization='global EEG window min/max including zero-valued unmapped Deep1010 rows, range [-1,1]',
        scale='unknown max_scale: auxiliary=0; not an EEG electrode',
        filters='no additional filters; reader resamples complete recordings to 256 Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        freeze_backbone=True)
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = BENDRClassifier(BENDR(**json.loads(payload['backbone_config_json'])), 2, freeze_backbone=True).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    if source_hashes != {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
