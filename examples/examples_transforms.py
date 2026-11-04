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

# Define separate signal and label callbacks.


def preprocess_fn(eeg, baseline=None, **kwargs) -> dict:
    eeg_transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),
    ])
    return {
        'eeg': eeg_transform(eeg),
        'baseline': eeg_transform(baseline) if baseline is not None else None,
    }


def transform_fn(eeg, baseline=None, **kwargs) -> dict:
    if baseline is not None:
        eeg = transforms.BaselineRemoval(baseline=baseline)(eeg)
    eeg_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return {
        'eeg': eeg_transform(eeg),
        'baseline': eeg_transform(baseline) if baseline is not None else None,
    }


def label_fn(info: dict, **kwargs) -> dict:
    label_transform = transforms.Compose([
        transforms.Select('label'),
    ])
    return {
        'label': label_transform(info),
    }


dataset = MemoryDataset(
    transform=transform_fn,
    label_transform=label_fn,
)
dataset.append({**preprocess_fn(eeg, baseline=eeg * 0.5),
                'info': {'label': 1}})
print(dataset[0]['eeg'].shape)
print(dataset[0]['label'])
