"""
Extract ordered EEGMat features with CBraModEmbedding
=====================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Wang et al. (2025), CBraMod. Source recordings use the Open
Data Commons Attribution License v1.0. Obtain complete subject pairs.
Labels background=0 and arithmetic=1 are saved only as source metadata;
no labels fit the backbone, resampling or feature extraction.

The EDF reader supplies microvolts and resamples whole recordings to 256 Hz.
Apply .3--75 Hz bandpass and 60 Hz notch MNE FIR filters to full recordings,
retain the original reference, reorder 19 channels into the author's order
and divide microvolts by 100. Use four-second/1024-sample windows. The model
resamples EACH of four one-second segments from 256 to 200 samples using
Fourier interpolation; periodic segment boundaries can ring. This differs
from preprocessing whole recordings directly at 200 Hz and is not a claim
of reproducing the author's pretraining or EEG-Arena protocol.

Freeze the complete backbone and keep it in evaluation mode. Flatten
[19 electrodes, 4 patches, H features] into 76H features per window. Save
rows in dataset order with source record/trial/subject metadata. The default
32-wide, two-block backbone is RANDOM. Optional --checkpoint safely loads
complete local canonical 200-wide, 12-block author weights. No dataset or
weight is downloaded. Random embeddings are not pretrained features.

Save the complete tensor state and constructor/preprocessing settings.
Rebuild CBraModEmbedding with those settings before strict state loading.
An output artifact contains EEG-derived features; keep it under the source
data-use terms. If fitting a classifier to these features, split subjects
first and fit feature statistics on training subjects only.

Run::

    python examples/examples_eegmat_cbramod_embeddings.py --root-path ./eegmat

Select new cache paths when source recordings or offline filters change.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import CBraMod, CBraModEmbedding


CHANNEL_ORDER = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=256,
        l_freq=.3, h_freq=75, verbose=False)
    return mne.filter.notch_filter(eeg, Fs=256, freqs=[60.], verbose=False)


def scale_microvolts(eeg):
    return eeg / 100.


def main():
    parser = argparse.ArgumentParser(description='Extract ordered EEGMat features with CBraModEmbedding.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-cbramod-embedding-19ch-256hz-4s')
    parser.add_argument('--output-path', default='./eegmat-cbramod-embedding.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive.')
    torch.manual_seed(42)
    indices = [MENTAL_ARITHMETIC_CHANNEL_LIST.index(name) for name in CHANNEL_ORDER]
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=1024, sampling_rate=256, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.PickElectrode(indices),
            transforms.Lambda(scale_microvolts), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if not subjects:
        raise ValueError('Use complete subject recording pairs.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    backbone_config = dict(patch_size=200, out_features=200, hid_channels=200, mlp_channels=800,
        depth=12, num_heads=8, dropout=.1, norm_eps=1e-5, position_kernel=(19, 7))
    if args.checkpoint:
        backbone = CBraMod.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **backbone_config)
        checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    else:
        backbone_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        backbone = CBraMod(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(num_electrodes=19, chunk_size=1024, num_patches=4,
                       flatten=True, freeze_encoder=True)
    model = CBraModEmbedding(backbone, **task_config).eval()
    with torch.no_grad():
        features = torch.cat([model(eeg).cpu() for eeg, _ in DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False)])
    if not torch.isfinite(features).all() or features.shape[0] != len(dataset):
        raise RuntimeError('Finite feature rows must match the unchanged dataset order.')
    metadata = json.loads(dataset.info.to_json(orient='records'))
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), backbone_config=backbone_config,
        task_config=task_config, embeddings=features, metadata=metadata,
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=CHANNEL_ORDER,
        sampling_rate=256, target_patch_sampling_rate=200, patch_size=200, patches=4,
        window_seconds=4, scale_microvolts=100.,
        filters=dict(bandpass_hz=(.3, 75.), notch_hz=60., method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        seed=42, torch_version=str(torch.__version__)), output)
    print(dict(windows=len(dataset), feature_shape=list(features.shape),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        artifact=str(output)))


if __name__ == '__main__':
    main()
