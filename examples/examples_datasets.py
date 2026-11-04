"""
Introduction to the datasets Module
===================================

Welcome to the guide on TorchEEG's ``datasets`` Module! This module provides you with various benchmark datasets for EEG-based emotion recognition, such as DEAP, DREAMER, SEED, MAHNOB, AMIGOS, and MPED. These datasets use a range of stimuli like music and videos to trigger emotional responses. Once the emotional experiences are recorded, participants label them using methods like valence-arousal dimensions or discrete emotion categories.

This guide will help you understand how to easily load and manipulate these datasets for your training needs, including signal segmentation, applying transformations, and more.

"""

######################################################################
# An Example of Emotion Recognition Dataset
# ----------------------------------------------
#
# To begin, you'll need to download the DEAP dataset from DEAP download link. Once you have the dataset, use the following code to load it. preprocess_fn computes features before storage; transform_fn prepares EEG and baseline on each read; label_fn constructs labels from info.
#

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_locations


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
        transforms.ToGrid(get_locations(
            DEAPDataset.channels, DEAPDataset.grid)),
        transforms.ToTensor(),
    ])
    return {
        'eeg': eeg_transform(eeg),
        'baseline': eeg_transform(baseline) if baseline is not None else None,
    }


def label_fn(info: dict, **kwargs) -> dict:
    label_transform = transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ])
    return {
        'label': label_transform(info),
    }


dataset = DEAPDataset(
    io_path=f'./deap',
    root_path='./data_preprocessed_python',
    num_worker=8,
    offline_transform=preprocess_fn,
    online_transform=transform_fn,
    label_transform=label_fn,
)

######################################################################
#
# You can get individual samples from the dataset like so:
#

print(dataset[0])

######################################################################
#
# This training view returns a dictionary with 'eeg' and 'label' fields.
#

######################################################################
#
# If you want to visualize the EEG signals, you can use the helper function as shown below:
#

import torch

from torcheeg.visualizers import plot_3d_tensor

img = plot_3d_tensor(torch.tensor(dataset[0]['eeg']))

######################################################################
#
# You can also refer to the document for more information on visualizing EEG signals
# via the ``plot_3d_tensor``, ``plot_feature_topomap``, ``plot_raw_topomap``, ``plot_signal``, and ``plot_adj_connectivity`` function.
#

######################################################################
# Important Notes
# ----------------------------------------------
#
# Use a new io_path after changing preprocessing. Reading transforms can be
# configured through online_transform without rebuilding the stored samples.
#
# .. code-block:: bash
#
#   !rm -rf ./deap
#

######################################################################
# If you remove all transforms and declare a dataset, you’ll find that it
# returns segmented EEG signals that haven’t undergone any preprocessing.
# Additionally, there is a dictionary representing all the
# meta-information corresponding to the EEG signal samples.
#

from torcheeg.datasets import DEAPDataset

dataset = DEAPDataset(
    io_path=f'./deap', root_path='./data_preprocessed_python')
print(dataset[0])

######################################################################
# Advanced Usage
# ----------------------------------------------
#
# Most emotion analysis datasets can be found in TorchEEG. Moreover,
# TorchEEG provides support for ``MoABB``, allowing access to
# motor-imagery-related datasets with the help of ``MoABB``.
#

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery

import torcheeg.datasets.moabb as moabb_dataset

dataset = BNCI2014001()
dataset.subject_list = [1, 2, 3]
paradigm = LeftRightImagery()


def preprocess_fn(eeg, baseline=None, **kwargs) -> dict:
    eeg_transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),
    ])
    return {
        'eeg': eeg_transform(eeg),
        'baseline': eeg_transform(baseline) if baseline is not None else None,
    }


def transform_fn(eeg, baseline=None, **kwargs) -> dict:
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


dataset = moabb_dataset.MOABBDataset(
    dataset=dataset, paradigm=paradigm, io_path='./moabb', offline_transform=preprocess_fn, online_transform=transform_fn, label_transform=label_fn)

######################################################################
# Custom Datasets
# ----------------------------------------------
#
# TorchEEG also supports custom datasets. You can place recorded EEG
# signal files in a folder following a specific rule, such as:
#
# ::
#
#    label01
#    |- sub01.edf
#    |- sub02.edf
#    label02
#    |- sub01.edf
#    |- sub02.edf
#
# Then, you can use ``FolderDataset`` to automatically access the
# corresponding EEG signal samples.
#

from torcheeg.datasets import FolderDataset

label_map = {'label01': 0, 'label02': 1}


def transform_fn(eeg, baseline=None, **kwargs) -> dict:
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
        transforms.Lambda(lambda x: label_map[x]),
    ])
    return {
        'label': label_transform(info),
    }


dataset = FolderDataset(
    io_path='./folder',
    root_path='./root_folder',
    structure='subject_in_label',
    num_channel=14,
    num_worker=4,
    online_transform=transform_fn,
    label_transform=label_fn,
)

######################################################################
# Alternatively, you can use a CSV file to specify more detailed
# meta-information for reading:
#
# ::
#
#    | subject_id | trial_id | label | file_path                 |
#    | ---------- | -------  | ----- | ------------------------- |
#    | sub1       | 0        | 0     | './data/label1/sub1.fif' |
#    | sub1       | 1        | 1     | './data/label2/sub1.fif' |
#    | sub1       | 2        | 2     | './data/label3/sub1.fif' |
#    | sub2       | 0        | 0     | './data/label1/sub2.fif' |
#    | sub2       | 1        | 1     | './data/label2/sub2.fif' |
#    | sub2       | 2        | 2     | './data/label3/sub2.fif' |
#

from torcheeg.datasets import CSVFolderDataset


def transform_fn(eeg, baseline=None, **kwargs) -> dict:
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


dataset = CSVFolderDataset(
    csv_path='./data.csv',
    num_worker=4,
    online_transform=transform_fn,
    label_transform=label_fn,
)

######################################################################
# By default, TorchEEG uses mne to read recorded EEG signals, but you can
# also specify your own file reading logic through ``read_fn``.
#

import mne


def default_read_fn(file_path, **kwargs):
    # Load EEG file
    raw = mne.io.read_raw(file_path)
    # Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1)
    # Return EEG data
    return epochs


def transform_fn(eeg, baseline=None, **kwargs) -> dict:
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


dataset = CSVFolderDataset(
    io_path='./csv_folder',
    csv_path='./data.csv',
    read_fn=default_read_fn,
    num_worker=4,
    online_transform=transform_fn,
    label_transform=label_fn,
)
