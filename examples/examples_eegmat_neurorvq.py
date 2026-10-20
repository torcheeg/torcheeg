"""
Probe complete external NeuroRVQ on EEGMat
==========================================

Use a separately obtained trusted KonstantinosBarmpas/NeuroRVQ checkout at
27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c and official Hub
ntinosbarmpas/NeuroRVQ revision d944b87f44ae0ba2923b2f10d0518f23f6803b76,
pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt.
Source and weights are CC-BY-NC4.0 and are not bundled with TorchEEG.
The later author microtuning source omits pretraining heads: this example
uses the complete historical implementation, all330states/32heads retained.
No files are downloaded and no checkpoint source code is executed.

The four raw branch features concatenate to800 dimensions and are averaged
over all patches; only a new classifier trains. Right-align temporal slots
and use the original zero-based electrode vocabulary. No patch truncation,
interpolation or channel remapping is allowed. The parameter-free feature
view adds no random normalization layers. Pretraining heads remain in state
and are callable separately via forward_masked; this probe does not train
RVQ targets or reproduce the original pretraining schedule.

Resample full EEGMat records500->200Hz, MNE FIR0.5--45Hz/notch50Hz, then
per-window/channel population z-score with std floor1e-6. This documented
teaching transfer normalization is not a published NeuroRVQ benchmark.
Keep all19EDFsource electrodes/reference. Last subject is held out, with no
parameter selection on test labels. Save and strictly restore every native
state, electrode vocabulary/order, classifier, preprocessing and predictions.
Cite Barmpas et al., NeuroRVQ, and Zyma et al. (2019), EEGMat,
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).

Run::

    python examples/examples_eegmat_neurorvq.py --root-path ./eegmat \
        --backend-root ./NeuroRVQ --checkpoint ./NeuroRVQ_EEG_foundation_model_v1.pt --epochs 1
"""
import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
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
from torcheeg.models import NeuroRVQ, NeuroRVQClassifier


VARIANT = 'eeg-v1-complete-external-backend'
SOURCE_REVISION = '27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c'


def build_neurorvq_backend(root, channel_names):
    """Construct a complete trusted author backend at the pretraining revision."""
    root = Path(root).resolve()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned NeuroRVQ pretraining revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    sys.path.insert(0, str(root))
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM
    from inference.modules.NeuroRVQ_EEG_FM_inference_modules import ch_names_global
    vocab = [name.decode() for name in ch_names_global]
    config = dict(n_patches=256, patch_size=200, in_chans=1, out_chans=8,
        num_classes=0, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.,
        qkv_bias=True, init_values=1e-5, n_global_electrodes=len(vocab),
        vocab_size=8192, use_as_encoder=True, use_for_pretraining=True)
    backend = NeuroRVQFM(**config, qk_norm=partial(nn.LayerNorm, eps=1e-6))
    # Parameter-free feature view retains all pretrained layers and 32 heads.
    backend.norm = nn.Identity()
    for index in range(1, 5):
        setattr(backend, 'fc_norm_'+str(index), nn.Identity())
        setattr(backend, 'head_'+str(index), nn.Identity())
    config['channel_vocab'] = vocab
    return backend, config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.5, h_freq=45., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples at200Hz.')
    eeg = np.asarray(eeg, dtype=np.float64)
    return ((eeg-eeg.mean(-1, keepdims=True))/eeg.std(-1, keepdims=True).clip(min=1e-6)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete NeuroRVQ EEG-v1 backend EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./eegmat-neurorvq-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-neurorvq.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_neurorvq_backend(args.backend_root, MENTAL_ARITHMETIC_CHANNEL_LIST)
    backbone = NeuroRVQ.from_pretrained(args.checkpoint, backend, MENTAL_ARITHMETIC_CHANNEL_LIST, backend_config['channel_vocab'])
    model = NeuroRVQClassifier(backbone, num_classes=2, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=800,
        num_channel=19, sampling_rate=200, io_path=args.cache_path, io_mode='pickle',
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
    config = dict(channel_names=list(backbone.channel_names), sampling_rate=backbone.sampling_rate,
        channel_vocab=list(backbone.channel_vocab))
    metadata = dict(variant=VARIANT, checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        reference='source EDF reference preserved; no re-referencing applied',
        sampling_rate=200, window_samples=800, input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR0.5--45Hz and50Hznotch on complete recordings AFTER resampling to200Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_backbone=True,
        weights_license='CC-BY-NC-4.0; separate source and checkpoint terms',
        backend_revision=SOURCE_REVISION, backend_config=backend_config, mne_version=str(mne.__version__),
        features='mean of all electrode/time patches, four raw branches concatenated to800 dimensions')
    artifact = dict(state_dict=model.state_dict(), backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored_config = json.loads(payload['backbone_config_json'])
    restored_backend, _ = build_neurorvq_backend(args.backend_root, restored_config['channel_names'])
    restored = NeuroRVQClassifier(NeuroRVQ(restored_backend, **restored_config), num_classes=2, freeze_backbone=True).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(payload['state_dict']['backbone.' + name], value, rtol=0, atol=0)
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
