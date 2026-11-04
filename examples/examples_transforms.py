"""
Transform arrays and samples
============================

Array transforms use positional input. Sample functions choose which fields to
transform; Compose applies functions in order.
"""

import numpy as np

from torcheeg import transforms
from torcheeg.datasets import MemoryDataset

# Transform an array.
eeg = np.random.default_rng(0).normal(size=(32, 128))
features = transforms.Compose([
    transforms.BandDifferentialEntropy(),
    transforms.ToTensor(),
])(eeg)
print(features.shape)

# Bind a baseline explicitly.
baseline = np.zeros_like(eeg)
corrected = transforms.BaselineRemoval(baseline=baseline)(eeg)

# Prepare a complete training sample in one function.


def preprocess_fn(sample: dict) -> dict:
    eeg_transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),
    ])
    return {
        **sample,
        'eeg': eeg_transform(sample['eeg']),
        'baseline': eeg_transform(sample['baseline']),
    }


def transform_fn(sample: dict) -> dict:
    eeg_transform = transforms.Compose([
        transforms.BaselineRemoval(baseline=sample['baseline']),
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.Select('label'),
    ])
    return {
        'eeg': eeg_transform(sample['eeg']),
        'label': label_transform(sample['info']),
    }


dataset = MemoryDataset(transform=transform_fn)
dataset.append(preprocess_fn(
    {'eeg': eeg, 'baseline': eeg * 0.5, 'info': {'label': 1}}))
print(dataset[0]['eeg'].shape)
print(dataset[0]['label'])
