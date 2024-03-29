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
# To begin, you'll need to download the DEAP dataset from DEAP download link. Once you have the dataset, use the following code to load it. This will read EEG signals and labels, apply offline transformations, and save them for easy access later on.
#

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=f'./deap',
    root_path='./data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)

######################################################################
#
# You can get individual samples from the dataset like so:
#

print(dataset[0])

######################################################################
#
# This will output a tuple, where the first element is the EEG signal and the second is its label.
#

######################################################################
#
# If you want to visualize the EEG signals, you can use the helper function as shown below:
#

import torch
from torcheeg.utils import plot_3d_tensor

img = plot_3d_tensor(torch.tensor(dataset[0][0]))

######################################################################
#
# You can also refer to the document for more information on visualizing EEG signals
# via the ``plot_3d_tensor``, ``plot_feature_topomap``, ``plot_raw_topomap``, ``plot_signal``, and ``plot_adj_connectivity`` function.
#

######################################################################
# Important Notes
# ----------------------------------------------
#
# If you make changes to the dataset's configuration, remember to clear the cache or specify a new io_path. Otherwise, only online transformations (online_transform and label_transform) will take effect:
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

dataset = DEAPDataset(io_path=f'./deap', root_path='./data_preprocessed_python')
print(dataset[0])

######################################################################
# Advanced Usage
# ----------------------------------------------
#
# Most emotion analysis datasets can be found in TorchEEG. Moreover,
# TorchEEG provides support for ``MoABB``, allowing access to
# motor-imagery-related datasets with the help of ``MoABB``.
#

import torcheeg.datasets.moabb as moabb_dataset

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery

dataset = BNCI2014001()
dataset.subject_list = [1, 2, 3]
paradigm = LeftRightImagery()
dataset = moabb_dataset.MOABBDataset(
    dataset=dataset,
    paradigm=paradigm,
    io_path='./moabb',
    offline_transform=transforms.Compose([transforms.BandDifferentialEntropy()
                                          ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([transforms.Select('label')]))

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
dataset = FolderDataset(io_path='./folder',
                        root_path='./root_folder',
                        structure='subject_in_label',
                        num_channel=14,
                        online_transform=transforms.ToTensor(),
                        label_transform=transforms.Compose([
                            transforms.Select('label'),
                            transforms.Lambda(lambda x: label_map[x])
                        ]),
                        num_worker=4)

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

dataset = CSVFolderDataset(csv_path='./data.csv',
                           online_transform=transforms.ToTensor(),
                           label_transform=transforms.Select('label'),
                           num_worker=4)

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


dataset = CSVFolderDataset(io_path='./csv_folder',
                           csv_path='./data.csv',
                           read_fn=default_read_fn,
                           online_transform=transforms.ToTensor(),
                           label_transform=transforms.Select('label'),
                           num_worker=4)