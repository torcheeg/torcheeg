"""
Transform arrays and samples
============================

Array transforms use positional input. Sample functions choose which fields to
transform; Compose applies functions in order.
"""

import numpy as np
import torch

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

# Apply the same array transform to multiple fields when needed.


def preprocess_sample(sample):
    feature = transforms.BandDifferentialEntropy()
    eeg = feature(sample['eeg'])
    baseline = feature(sample['baseline'])
    return {**sample, 'eeg': transforms.BaselineRemoval(baseline)(eeg),
            'baseline': baseline}


def prepare_training(sample):
    return {'eeg': transforms.ToTensor()(sample['eeg']),
            'labels': torch.tensor(sample['info']['label'])}


dataset = MemoryDataset()
dataset.append({'eeg': eeg, 'baseline': eeg * 0.5, 'info': {'label': 1}})
training = dataset.with_transform(
    preprocess_sample).with_transform(prepare_training)
equivalent = dataset.with_transform(transforms.Compose([
    preprocess_sample,
    prepare_training,
]))
torch.testing.assert_close(training[0]['eeg'], equivalent[0]['eeg'])
print(training[0]['eeg'].shape)
