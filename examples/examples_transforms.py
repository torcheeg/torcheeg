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


def transform_fn(sample):
    eeg_transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),
        transforms.BaselineRemoval(
            baseline=transforms.BandDifferentialEntropy()(sample["baseline"]),
        ),
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.Select("label"),
    ])
    return {
        "eeg": eeg_transform(sample["eeg"]),
        "label": label_transform(sample["info"]),
    }


dataset = MemoryDataset()
dataset.append({'eeg': eeg, 'baseline': eeg * 0.5, 'info': {'label': 1}})
training = dataset.with_transform(transform_fn)
print(training[0]['eeg'].shape)
print(training[0]['label'])
