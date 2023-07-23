"""
Introduction to the Datasets Module
===================================

Introduction to the Datasets Module The datasets module supports a
variety of benchmark datasets for EEG-based emotion recognition,
including DEAP, DREAMER, SEED, MAHNOB, AMIGOS, and MPED datasets. These
datasets use diverse stimuli such as music and videos to induce
emotional responses in participants. Participants’ EEG signals are
recorded during these stimuli, and they are then asked to label their
emotional experiences using methods like the valence-arousal dimension
or discrete emotion categories.

The datasets module allows the configuration to load a specific
dataset and segment the EEG signals into, for example, one-second-long
windows. Developers can then access these EEG segments and their
corresponding meta-information such as stimulus emotions, trial
information, session details, subject descriptions, and stimulus
multimedia for training purposes.

"""


######################################################################
# An example of this is creating a DEAPDataset. First, you need to
# download the DEAP dataset from
# https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html. Then, you
# can create a dataset using the code below. The primary function of this
# code is to read EEG signals and corresponding labels from the
# ``root_path``, segment the EEG signals as samples, perform
# ``offline_transform`` and save them to ``io_path``. During each index,
# it automatically reads the processing results from ``io_path`` and
# performs ``online_transform`` and label_transform to obtain
# corresponding samples.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants.emotion_recognition.deap import \
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
# You can access the corresponding samples in the dataset by indexing. It
# returns a tuple where the first object represents the EEG signal, and
# the second object represents the label.
# 

print(dataset[0])


######################################################################
# You can visualize the EEG signals in the samples through the helper
# function.
# 

import torch
from torcheeg.utils import plot_2d_tensor

img = plot_2d_tensor(torch.tensor(dataset[0][0]))


######################################################################
# When you wish to use a new configuration to read datasets, please delete
# the original cache or modify the ``io_path`` to save the cache in a new
# location. Otherwise, changes made to configurations like
# ``offline_transform`` will not take effect. Only ``online_transform``
# and ``label_transform`` will remain active, functioning as online
# processing.
# 

!rm -rf ./deap


######################################################################
# If you remove all transforms and declare a dataset, you’ll find that it
# returns segmented EEG signals that haven’t undergone any preprocessing.
# Additionally, there is a dictionary representing all the
# meta-information corresponding to the EEG signal samples.
# 

from torcheeg.datasets import DEAPDataset

dataset = DEAPDataset(
    io_path=f'./deap',
    root_path='./data_preprocessed_python')
print(dataset[0])


######################################################################
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
dataset = moabb_dataset.MOABBDataset(dataset=dataset,
                       paradigm=paradigm,
                       io_path='./moabb',
                       offline_transform=transforms.Compose(
                           [transforms.BandDifferentialEntropy()]),
                       online_transform=transforms.ToTensor(),
                       label_transform=transforms.Compose([
                           transforms.Select('label')
                       ]))


######################################################################
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