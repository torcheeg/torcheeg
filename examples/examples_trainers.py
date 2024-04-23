"""
Introduction to the trainers Module
===================================

Welcome to the guide on TorchEEG's ``trainers`` Module! This module provides you with a suite of trainers, built on Pytorch-lightning, for
model training. These trainers are designed to handle a wide array of
models, ranging from discriminative to generative ones, and are equipped
with capabilities for contrast learning and fine-tuning. They are also
capable of running on various hardware configurations, from single CPUs,
single GPUs, to multiple GPUs. You also have the flexibility to tailor
these trainers according to your specific needs by extending the
existing ones.

"""


######################################################################
# Discriminative Models with ClassifierTrainer
# ----------------------------------------------
#
# The simplest yet highly effective method for discriminative models
# involves utilizing a classification loss function, such as cross-entropy
# for EEG signal recognition training, encapsulated in the
# ClassifierTrainer algorithm.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT
from torch.utils.data import DataLoader
from torcheeg.models import CCNN

from torcheeg.trainers import ClassifierTrainer

import pytorch_lightning as pl

dataset = DEAPDataset(
    io_path=f'./examples_trainers_1/deap',
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

k_fold = KFoldGroupbyTrial(n_splits=10,
                           split_path='./examples_trainers_1/split',
                           shuffle=True,
                           random_state=42)

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                devices=1,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_trainers_1/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')


######################################################################
# Within ClassifierTrainer, you can specify the device and the number of
# devices to use. In the trainer.fit function, you can pass any parameters
# supported by the Trainer class in pytorch_lightning. If you wish to
# modify the training procedure, you can do so by extending the trainers:
# 

class MyClassifierTrainer(ClassifierTrainer):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # your loss function
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss


######################################################################
# By referring to the related documents, you can also learn about the
# evaluation metrics supported in the trainers module. By making a few
# adjustments, you can get the trainer to report metrics like accuracy and
# f1 score.
# 

trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                metrics=['accuracy', 'recall', 'precision', 'f1score'],
                                accelerator="gpu")


######################################################################
# Domain Adaptation Methods
# ----------------------------------------------
#
# A challenging aspect of EEG-based emotion recognition is the
# cross-subject problem. Even when evoked by the same stimulus, the
# distribution of EEG signal patterns among different individuals may
# undergo distributional shifts. This phenomenon negatively impacts the
# performance of the trained models when applied to unknown subjects. The
# trainers module addresses this issue by providing a host of domain
# adaptation algorithms. These algorithms use EEG samples from known
# subjects (source domain) and unknown subjects for testing (target
# domain). They optimize models with specific loss functions and training
# strategies to extract domain-invariant features or transfer knowledge
# from the source to the target domain. A range of cross-domain trainers
# like CORALTrainer, ADATrainer, DANNTrainer, DDCTrainer, and DANTrainer
# are available to handle different application scenarios and assist users
# in dealing with cross-domain problems across diverse models and
# datasets.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT
from torch.utils.data import DataLoader
from torcheeg.models import CCNN

from torcheeg.trainers import CORALTrainer

import pytorch_lightning as pl

dataset = DEAPDataset(
    io_path=f'./examples_trainers_2/deap',
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

k_fold = LeaveOneSubjectOut(split_path='./examples_trainers_2/split')


class Extractor(CCNN):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        return x


class Classifier(CCNN):
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    source_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    extractor = Extractor(in_channels=5, num_classes=2)
    classifier = Classifier(in_channels=5, num_classes=2)

    trainer = CORALTrainer(extractor=extractor,
                                classifier=classifier,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=0.0,
                                accelerator='gpu')
    trainer.fit(source_loader,
                target_loader,
                target_loader,
                max_epochs=50,
                default_root_dir=f'./examples_trainers_2/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(target_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')


######################################################################
# Generative Models
# ----------------------------------------------
#
# For generative models, we provide trainers for models including VAE,
# GAN, Normalizing Flow, and Diffusion Model. These trainers aim to train
# models to generate EEG signals that closely mimic the real distribution.
# We also provide conditional versions of these trainers, enabling the use
# of categories as conditions for category-driven EEG sample generation.
# 

from torcheeg.trainers import CDDPMTrainer
from torcheeg.models import BCUNet

model = BCUNet(in_channels=4)
trainer = CDDPMTrainer(model, accelerator='gpu')

from torcheeg.trainers import CWGANGPTrainer
from torcheeg.models import BCGenerator, BCDiscriminator

g_model = BCGenerator(in_channels=128)
d_model = BCDiscriminator(in_channels=4)

trainer = CWGANGPTrainer(g_model,
                              d_model,
                             accelerator='gpu')

from torcheeg.trainers import CGlowTrainer
from torcheeg.models import BCGlow

model = BCGlow(in_channels=4)
trainer = CGlowTrainer(model, accelerator='gpu')


######################################################################
# TorchEEG supports common evaluation metrics for generative models, such
# as FID. To use these metrics, you need to provide additional parameters
# to the Trainer like metric_extractor, metric_classifier, and
# metric_num_features. For details, please refer to the related documents.
# Here is an example:
# 

import torch.nn as nn

class Extractor(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        return x


class Classifier(nn.Module):

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(64, 128, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(128, 256, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
                                   nn.Conv2d(256, 64, kernel_size=4, stride=1),
                                   nn.ReLU())

        self.lin1 = nn.Linear(9 * 9 * 64, 1024)
        self.lin2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
g_model = BCGenerator(in_channels=128)
d_model = BCDiscriminator(in_channels=4)

extractor = Extractor()
classifier = Classifier()
# you may need to load state dict from your trained extractor, classifier

trainer = CWGANGPTrainer(g_model,
                              d_model,
                              metric_extractor=extractor,
                              metric_classifier=classifier,
                              metric_num_features=9 * 9 * 64,
                              metrics=['fid'],
                             accelerator='gpu')