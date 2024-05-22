"""
Introduction to the transforms Module
=====================================

Welcome to the guide on TorchEEG's ``transforms`` Module! This module provides you with toolset designed for the transformation and preprocessing of EEG signals and related metadata.

EEG transforms concentrate on EEG preprocessing, feature extraction,
data conversion, and data augmentation for deep learning models. They
typically accept various parameters for instantiation, and the
instantiated transformers can be used as functions to process EEG
signals. They accept 'eeg' and 'baseline' as keyword arguments and
return a dictionary, in which 'eeg' corresponds to the processed EEG
signal, while 'baseline' corresponds to the processed baseline signal.

"""

import numpy as np
from torcheeg import transforms

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

t = transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
eeg = t(eeg=np.random.randn(32, 128))['eeg']
print(eeg.shape)

eeg = t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg']
print(eeg.shape)

######################################################################
# When it comes to preprocessing, EEG transforms aid in the elimination of
# noise from EEG signals. A prime example of this is the BaselineRemoval,
# which employs the baseline signal prior to the stimulus to remove
# stimulus-independent fluctuations from the processed signal.
#

t = transforms.BaselineRemoval()

eeg = t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg']
print(eeg.shape)

######################################################################
# For feature extraction, EEG transforms take charge of creating
# emotion-discriminative features usually employed in emotion recognition.
# One such example is the BandDifferentialEntropy, which gauges the
# differential entropy of the frequency subbands within the EEG signal.
#

eeg = np.random.randn(32, 128)
transformed_eeg = transforms.BandDifferentialEntropy()(eeg=eeg)['eeg']

######################################################################
# In terms of data conversion, EEG transforms play a pivotal role in
# shaping EEG signals into time series, 3D grids, or graph networks.
# Examples of these include ToGrid, ToInterpolatedGrid, ToG, and
# ToDynamicG.
#

from torcheeg.datasets.constants import DEAP_ADJACENCY_MATRIX
from torcheeg.transforms.pyg import ToG

eeg = np.random.randn(32, 128)
transformed_eeg = ToG(DEAP_ADJACENCY_MATRIX)(eeg=eeg)['eeg']
print(transformed_eeg)

######################################################################
# Regarding data augmentation, transforms such as RandomMask apply random
# transformations to the EEG signals, effectively augmenting the data
# volume.
#

import torch

eeg = torch.randn(32, 128)
transformed_eeg = transforms.RandomMask(p=1.0)(eeg=eeg)['eeg']
print(transformed_eeg.shape)

######################################################################
# Metadata transforms interact with the metadata associated with EEG
# signals, generating ancillary information or labels needed for models.
# An example is Binary, which transforms continuous emotional valence or
# arousal into binary labels.
#

info = {'valence': 4.5, 'arousal': 5.5, 'subject_id': 7}
transformed_label = transforms.Select(key='valence')(y=info)['y']
print(transformed_label)

transformed_label = transforms.Binary(threshold=5.0)(y=transformed_label)['y']
print(transformed_label)

######################################################################
# Process transforms bring several transformers together, crafting the
# overall processes. For instance, Compose allows the output of one
# transformation to be the input of the next, thus chaining multiple
# transforms together.
#

from torcheeg import transforms

t = transforms.Compose([
    transforms.BandDifferentialEntropy(),
    transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
])
eeg = t(eeg=np.random.randn(32, 128))['eeg']
print(eeg.shape)

######################################################################
# It's worth noting that unless specifically required, transformers before
# BaselineRemoval should be set with apply_to_baseline=True, to ensure the
# baseline signal and the experiment signal undergo the same
# transformations. Of course, you don't need to do this if BaselineRemoval
# is not needed.
#

t = transforms.Compose([
    transforms.BandDifferentialEntropy(apply_to_baseline=True),
    transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True),
    transforms.BaselineRemoval()
])
eeg = t(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg']
print(eeg.shape)
