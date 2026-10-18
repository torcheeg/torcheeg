"""
Extract ordered EEGMat features with STEEGFormerEmbedding
=========================================================

Use the public EEG During Mental Arithmetic Tasks database:
https://physionet.org/content/eegmat/1.0.0/. Cite Zyma et al. (2019),
PhysioNet and Yang et al. (ICLR 2026), ST-EEGFormer. Source recordings use
Open Data Commons Attribution License v1.0. No data or weights are downloaded.

Supply complete subject pairs. Save one mean-pooled H-wide embedding per
window in unchanged dataset order, together with every source metadata row.
There is no learned task head or label-fitted preprocessing. Labels are
retained only as source metadata. Split subjects before fitting subsequent
feature statistics or classifiers.

The reader supplies microvolts and resamples complete recordings to 128 Hz.
Filter complete recordings at .5--45 Hz with MNE FIR and take four-second,
512-sample windows. Preserve all 19 source channels, their order and EEG
reference. The wrapper standardizes each channel/window using population
standard deviation (ddof=0, floor 1e-6 microvolts). No statistics are fitted
across subjects. Optional whole-window Fourier resampling in the wrapper
is an identity at this native rate; 32 complete 16-sample patches remain.

Freeze every encoder parameter and keep it in evaluation mode. Mean-pool
PRE-NORM patch tokens excluding CLS, as in the official downstream path.
The default 32-wide, two-block encoder is RANDOM. Optional --checkpoint
loads a complete local canonical small encoder (512-wide, eight blocks).
For the native author training file add --allow-native-checkpoint to opt
into scoped argparse.Namespace metadata and recognized MAE decoder removal,
with weights_only=True. Random outputs are not pretrained representations.
This teaching protocol does not reproduce the author's benchmark or EEG-Arena.

Save complete tensor state, constructor/preprocessing/channel settings and
all ordered features and metadata. Rebuild STEEGFormer and its task wrapper with the
saved configurations, then load state strictly. No optimizer state is saved.
Keep EEG-derived artifacts under the source data-use terms. Select new
cache/split paths when input recordings or offline preprocessing change.

Run::

    python examples/examples_eegmat_steegformer_embeddings.py --root-path ./eegmat
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
from torcheeg.models import STEEGFormer, STEEGFormerEmbedding


def filter_recording(eeg):
    return mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=128,
        l_freq=.5, h_freq=45., verbose=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Extract ordered EEGMat features with STEEGFormerEmbedding.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./eegmat-steegformer-embedding-19ch-128hz-4s')
    parser.add_argument('--output-path', default='./eegmat-steegformer-embedding.pt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--allow-native-checkpoint', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.allow_native_checkpoint and not args.checkpoint:
        parser.error('native-checkpoint mode requires --checkpoint.')
    if args.batch_size < 1:
        parser.error('batch-size must be positive.')
    torch.manual_seed(42)
    dataset = MentalArithmeticDataset(root_path=args.root_path, io_path=args.cache_path,
        io_mode='pickle', num_channel=19, chunk_size=512, sampling_rate=128, num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if not subjects:
        raise ValueError('Use complete subject recording pairs.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    backbone_config = dict(patch_size=16, hid_channels=512, mlp_channels=2048, depth=8, num_heads=8,
        num_channel_embeddings=145, max_patches=511, qkv_bias=True, dropout=0., attention_dropout=0.,
        drop_path_rate=0., norm_eps=1e-6)
    if args.checkpoint:
        backbone = STEEGFormer.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix,
            allow_namespace=args.allow_native_checkpoint, exclude_decoder=args.allow_native_checkpoint, **backbone_config)
        checkpoint_sha256 = file_sha256(args.checkpoint)
    else:
        backbone_config.update(hid_channels=32, mlp_channels=128, depth=2, num_heads=4)
        backbone = STEEGFormer(**backbone_config)
        checkpoint_sha256 = None
    task_config = dict(channel_names=MENTAL_ARITHMETIC_CHANNEL_LIST, chunk_size=512, sampling_rate=128,
        pooling='mean', normalize_signal=True, std_correction=0, std_floor=1e-6,
        normalize_features=False, freeze_encoder=True)
    model = STEEGFormerEmbedding(backbone, **task_config).eval()
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
        source_channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST,
        sampling_rate=128, native_sampling_rate=128, patch_size=16, patches=32, window_seconds=4,
        normalization=dict(axis='channel-wise time before resampling', ddof=0, std_floor_microvolts=1e-6),
        filters=dict(bandpass_hz=(.5, 45.), method='MNE FIR on full recordings after resampling'),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        seed=42, torch_version=str(torch.__version__)), output)
    print(dict(windows=len(dataset), feature_shape=list(features.shape),
        checkpoint_loaded=bool(args.checkpoint), checkpoint_sha256=checkpoint_sha256,
        artifact=str(output)))


if __name__ == '__main__':
    main()
