"""
Introduction to the models Module
====================================

Welcome to the guide on TorchEEG's ``models`` Module! This module provides you with a variety of both discriminative and generative models. Each model has been meticulously replicated from its respective research studies, offering a toolkit for EEG signal analysis.

"""


######################################################################
# Discriminative Models
# ----------------------------------------------
#
# Regarding discriminative models, the models module houses
# implementations of Convolutional Neural Networks (CNNs), such as EEGNet
# and TSception. These models create a variety of spatial, temporal, or
# spectral tensor representations of EEG signals and execute local pattern
# analysis via convolution.
# 

import torch
from torcheeg.models.cnn import TSCeption

eeg = torch.randn(1, 1, 28, 512)
model = TSCeption(num_classes=2,
                  num_electrodes=28,
                  sampling_rate=128,
                  num_T=15,
                  num_S=15,
                  hid_channels=32,
                  dropout=0.5)
pred = model(eeg)


######################################################################
# To understand how to preprocess datasets into formats required by the
# models, please refer to the corresponding documentation of each model.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    chunk_size=512,
                    num_baseline=1,
                    baseline_chunk_size=512,
                    offline_transform=transforms.Compose([
                        transforms.PickElectrode(transforms.PickElectrode.to_index_list(
                        ['FP1', 'AF3', 'F3', 'F7',
                        'FC5', 'FC1', 'C3', 'T7',
                        'CP5', 'CP1', 'P3', 'P7',
                        'PO3','O1', 'FP2', 'AF4',
                        'F4', 'F8', 'FC6', 'FC2',
                        'C4', 'T8', 'CP6', 'CP2',
                        'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
                        transforms.To2d()
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))

model = TSCeption(num_classes=2,
                  num_electrodes=28,
                  sampling_rate=128,
                  num_T=15,
                  num_S=15,
                  hid_channels=32,
                  dropout=0.5)
x = dataset[0][0]
x = torch.unsqueeze(x,dim=0)
print(model(x))


######################################################################
# The module also includes Recurrent Neural Networks (RNNs) like GRU and
# LSTM. These models view EEG signals as multivariate time series data and
# construct recurrent modules for emotion decoding.
# 

from torcheeg.models import GRU

model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

eeg = torch.randn(2, 32, 128)
pred = model(eeg)


######################################################################
# In the field of Graph Neural Networks (GNNs), the models module
# incorporates renowned networks like DGCNN, RGNN, and LGGNet. These
# models aim to analyze the functional connections between electrodes by
# depicting them as a graph network and designing graph convolution
# kernels.
# 

from torcheeg.models import DGCNN

eeg = torch.randn(1, 62, 200)
model = DGCNN(in_channels=200, num_electrodes=62, hid_channels=32, num_layers=2,num_classes=2)
pred = model(eeg)


######################################################################
# In recent years, the increasing popularity of Transformer-based models,
# such as EEG-ConvTransformer, has been recognized in the models module.
# These models predominantly utilize various self-attention mechanisms to
# analyze electrode correlations, delivering valuable insights.
# 

from torcheeg.models import SimpleViT

eeg = torch.randn(1, 128, 9, 9)
model = SimpleViT(chunk_size=128, t_patch_size=32, s_patch_size=(3, 3), num_classes=2)
pred = model(eeg)


######################################################################
# Some studies have shown that attention based models have achieved good 
# classification performance in EEG, such as Altaheri et al.'s ATCNet, 
# which uses moving windows in the model structure and utilizes 
# multiheadattention to process data within the window.This model achieved 
# excellent results in data set 2a of the BCI Competition IV.
#

from torcheeg.models import ATCNet
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms

dataset = BCICIV2aDataset(io_path=f'./bciciv_2a',
                              root_path='./BCICIV_2a_mat',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('label'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]))                             
model = ATCNet(num_classes=4,
               num_windows=3,
               in_channels=22,
               chunk_size=1750)
x = dataset[0][0]
pred = model(x)


######################################################################
# Generative Models
# ----------------------------------------------
#
# We also provide a variety of generative models known for their
# impressive advancements in computer vision, natural language processing,
# and other domains. When applied to EEG analysis, four categories of
# generative models are offered as sturdy benchmarks for researchers. The
# Generative Adversarial Network (GAN), for instance, WGAN-GP, includes a
# generator and discriminator. The discriminator learns to distinguish
# between real and generated signals, and the generator is trained to
# approximate accurate signals through adversarial training against the
# discriminator.
# 

from torcheeg.models import BCGenerator, BCDiscriminator

g_model = BCGenerator(in_channels=128, num_classes=3)
d_model = BCDiscriminator(in_channels=4, num_classes=3)
z = torch.normal(mean=0, std=1, size=(1, 128))
y = torch.randint(low=0, high=3, size=(1, ))
fake_X = g_model(z, y)
disc_X = d_model(fake_X, y)


######################################################################
# The Variational Auto-Encoder (VAE), such as Beta VAE, equipped with an
# encoder and decoder, maps observed EEG signals into a latent space using
# the encoder and then employs the decoder to reproduce EEG signals.
# 

from torcheeg.models import BCEncoder, BCDecoder

encoder = BCEncoder(in_channels=4, num_classes=3)
decoder = BCDecoder(in_channels=64, out_channels=4, num_classes=3)
y = torch.randint(low=0, high=3, size=(1, ))
mock_eeg = torch.randn(1, 4, 9, 9)
mu, logvar = encoder(mock_eeg, y)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = eps * std + mu
fake_X = decoder(z, y)


######################################################################
# Normalizing Flow, for example, Glow, offers a series of invertible
# transformations. It learns the sequence of reversible transformations to
# convert EEG signals into latent variables and utilizes the reverse of
# the flow function to return to samples for generation.
# 

import torch.nn.functional as F

import torch
from torcheeg.models import BCGlow

model = BCGlow(num_classes=2)
# forward to calculate loss function
mock_eeg = torch.randn(2, 4, 32, 32)
y = torch.randint(0, 2, (2, ))

y = y.float()
nll_loss, y_logits = model(mock_eeg, y)
loss = nll_loss.mean() + F.cross_entropy(y_logits, y)

# sample a generated result
y = y.to(torch.int64)
fake_X = model.sample(y, temperature=1.0)


######################################################################
# The Diffusion Model, such as DDPM, introduces a sequential corruption of
# observed data with increasing noise and learns to reverse this process.
# The generation process inverts this diffusion process, starting with
# white noise and gradually denoising it into corresponding observed EEG
# signals.
# 

from torcheeg.models import BCUNet

unet = BCUNet(num_classes=2)
mock_eeg = torch.randn(2, 4, 9, 9)
t = torch.randint(low=1, high=1000, size=(2, ))
y = torch.randint(low=0, high=2, size=(1, ))
fake_X = unet(mock_eeg, t, y)


######################################################################
# Eegtorch provides other types of models, such as eegfusenet, which combines 
# the functions of EEG encoding and generating new samples. At the same time, 
# eefusenet is an unsupervised learning model that can extract deep feature 
# encoding from input EEG signals and ultimately generate similar new samples. 
# Eegfusenet uses a approach similar to traditional gan models to identify 
# whether samples is real: EFDiscriminator, which ultimately improves the quality 
# of sample generation through eegfuset after adversarial training.
#

from torcheeg.models import EEGfuseNet,EFDiscriminator

fusenet = EEGfuseNet(32,16,1,1,384)
eeg = torch.randn(2, 32, 384) 
# simply input the EEG signal to output generated samples and deep fusion codes
fake_X,deep_code = fusenet(eeg)

discriminator = EFDiscriminator(32,1,1,384)
p_real = discriminator(eeg)
p_fake = discriminator(fake_X)