"""
Extract EEGPT embeddings from BCIC-IV-2B
========================================

Read BCI Competition IV 2b through MOABB using C3, Cz, C4, a 4--38 Hz
bandpass, and resampling from 250 to 256 Hz. Select 0--4 seconds relative
to the imagery interval (3--7 after events), one 1024-sample window per
retained epoch. Source arrays are microvolts; standardize each channel
within each window. Cite Leeb et al. (2007), Tangermann et al. (2012),
MOABB and Wang et al. (2024), EEGPT. Obtain recordings under their terms:
https://bnci-horizon-2020.eu/database/data-sets.

The default small encoder (width32, depth2, four heads, two summary tokens)
is randomly initialized. Its features are NOT pretrained EEGPT features.
Provide --checkpoint for a complete local official-large encoder state;
use --checkpoint-prefix '' for a plain tensor dictionary. No automatic
weight download or unsafe deserialization is performed.

EEGPTEmbedding keeps the encoder frozen and in evaluation mode. Save
flattened features with one metadata row per window in the same order.
No labels, population statistics, classifier or test-set tuning are used.
Downstream evaluation must create its own subject-separated training split
before fitting a feature scaler or a task head.

Run::

    python examples/examples_bciciv2b_eegpt_embeddings.py --download-path ./bciciv2b-raw --output-path ./embeddings.pt

Keep raw downloads outside the cache. The output contains EEG-derived
features and source metadata; keep it under the dataset's data-use terms.
"""

import argparse
import json
from pathlib import Path

import torch
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets.moabb import MOABBDataset
from torcheeg.models import EEGPT, EEGPTEmbedding


def main():
    parser = argparse.ArgumentParser(description='Extract EEGPT embeddings from BCIC-IV-2B.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--download-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-eegpt-embedding-3ch-256hz-4s')
    parser.add_argument('--output-path', default='./bciciv2b-eegpt-embeddings.pt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint', help='Local complete official-large encoder checkpoint.')
    parser.add_argument('--checkpoint-prefix', default='target_encoder.')
    args = parser.parse_args()
    subjects = sorted(set(args.subjects))
    if len(subjects) < 1 or any(subject < 1 or subject > 9 for subject in subjects):
        parser.error('Select subject IDs between 1 and 9.')
    if args.batch_size < 1:
        parser.error('batch-size must be positive.')

    torch.manual_seed(42)
    source = BNCI2014_004()
    source.subject_list = subjects
    paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                tmin=0, tmax=4, resample=256)
    dataset = MOABBDataset(dataset=source, paradigm=paradigm,
        download_path=args.download_path, io_path=args.cache_path, io_mode='pickle',
        chunk_size=1024, overlap=0, num_worker=0,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    if set(dataset.info.subject_id) != set(subjects):
        raise ValueError('The processed source must contain every selected subject.')
    for subject in subjects:
        labels = set(dataset.info.loc[dataset.info.subject_id == subject, 'label'])
        if labels != {'left_hand', 'right_hand'}:
            raise ValueError(f'Subject {subject} must contain both imagery classes.')

    parameters = dict(num_electrodes=3, chunk_size=1024, channel_names=['C3', 'CZ', 'C4'])
    if args.checkpoint:
        encoder = EEGPT.from_pretrained(args.checkpoint, prefix=args.checkpoint_prefix, **parameters)
    else:
        # A small, randomly initialized encoder keeps this teaching run inexpensive.
        encoder = EEGPT(hid_channels=32, depth=2, num_heads=4, embed_num=2, **parameters)
    model = EEGPTEmbedding(encoder, freeze_encoder=True).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        features = torch.cat([model(eeg).cpu() for eeg, _ in loader])
    if not torch.isfinite(features).all() or features.shape[0] != len(dataset):
        raise RuntimeError('Feature rows must be finite and match the dataset order.')
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(dataset.info.to_json(orient='records'))
    torch.save({'embeddings': features, 'metadata': metadata,
                'checkpoint_loaded': bool(args.checkpoint), 'channels': ['C3', 'CZ', 'C4'],
                'sampling_rate': 256, 'chunk_size': 1024}, output)
    print({'windows': len(dataset), 'feature_shape': list(features.shape),
           'checkpoint_loaded': bool(args.checkpoint), 'artifact': str(output)})


if __name__ == '__main__':
    main()
