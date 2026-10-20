"""
Probe pretrained BIOT on EEGMat
===============================

Load one of the three official BIOT encoder checkpoints, freeze it, and train
its native ELU/linear task head on background=0 versus mental arithmetic=1.
Use the public EEG During Mental Arithmetic Tasks database (Zyma et al., 2019):
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution License v1.0).
Cite that source and BIOT (Yang et al., NeurIPS 2023).

The reader resamples complete recordings from 500 to 200 Hz and returns
microvolts. Build the 16 PREST bipolar derivations in the exact pretrained
order (T3/T4/T5/T6 are T7/T8/P7/P8 aliases), then convert to volts. Normalize
each channel of each four-second window by its own linear-interpolated 95th
percentile absolute amplitude plus 1e-8 volts, matching the author's loader
normalization. No additional filtering, clipping or learned scaling occurs.
For both 18-token variants use the first 16 token IDs: the source lacks
A1/A2, so C3-A2 and C4-A1 are explicitly omitted rather than synthesized.

Hold out the last subject in sorted ID order, retain all its windows for
final evaluation, and train on all other subjects. Run a fixed epoch count;
never select epochs/settings using test labels. A one-epoch teaching run
verifies functionality, not benchmark performance or clinical utility.
The original pretraining/task-specific protocols differ from this example.

Download a checkpoint yourself from https://github.com/ycq091044/BIOT at
revision d138e32634e52ae9fa6ec98ac9c4087b14ca869a and check its SHA256 in
the BIOT guide. This script requires a genuine full-size local checkpoint;
it does not download data/weights or silently substitute random weights.
Save the complete model, constructor configuration, preprocessing, source
hashes, split, logits and labels. Metrics use JSON metadata for compatibility
with older weights-only readers. Strictly restore and check logits before
reporting the run.

Run::

    python examples/examples_eegmat_biot.py --root-path ./eegmat \\
        --checkpoint ./EEG-PREST-16-channels.ckpt --variant prest-16ch \\
        --cache-path ./cache-eegmat-biot --output-path ./biot-probe.pt --epochs 1

Use a fresh cache path when source or preprocessing changes. The standalone
workflow uses the public BIOT and BIOTClassifier classes and a PyTorch loop.
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
from torcheeg.models import BIOT, BIOTClassifier


BIPOLAR_PAIRS = [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
                 ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
                 ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
                 ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')]


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples at 200 Hz.')
    channels = MENTAL_ARITHMETIC_CHANNEL_LIST
    bipolar = np.stack([eeg[channels.index(left)] - eeg[channels.index(right)]
                        for left, right in BIPOLAR_PAIRS]).astype(np.float64) * 1e-6
    # NumPy's default quantile method is linear, including old supported NumPy.
    scale = np.quantile(np.abs(bipolar), .95, axis=-1, keepdims=True)
    return (bipolar / (scale + 1e-8)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen full-size pretrained BIOT EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--variant', choices=sorted(BIOT.variants), default='shhs-prest-18ch')
    parser.add_argument('--cache-path', default='./eegmat-biot-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-biot.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backbone = BIOT.from_pretrained(args.checkpoint, variant=args.variant)
    model = BIOTClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=800,
        num_channel=19, sampling_rate=200, io_path=args.cache_path, io_mode='pickle',
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
    config = dict(num_electrodes=BIOT.variants[args.variant], hid_channels=256, num_heads=8,
        depth=4, n_fft=200, hop_length=100, dropout=.1, attn_dropout=.2)
    metadata = dict(variant=args.variant, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        source_sha256=source_hashes, source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        bipolar_pairs=BIPOLAR_PAIRS, channel_ids=list(range(16)), sampling_rate=200,
        window_samples=800, input_unit='microvolts from MentalArithmeticDataset',
        normalization='bipolar microvolts * 1e-6 / (linear q95(abs(bipolar volts)) + 1e-8 volts)',
        filters='no additional filters; reader resamples complete recordings to 200 Hz',
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
    restored = BIOTClassifier(BIOT(**json.loads(payload['backbone_config_json'])), 2, freeze_backbone=True).eval()
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
