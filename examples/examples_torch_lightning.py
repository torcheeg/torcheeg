"""Training models with Pytorch-Lightning
======================================
In this case, we introduce how to use TorchEEG and Pytorch-Lightning to train a Continuous Convolutional Neural Network (CCNN) on the DEAP dataset for emotion classification.
"""

import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFold
from torcheeg.models import CCNN

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Set the random number seed in all modules to guarantee the same result when running again.

seed_everything(42)

###############################################################################
# Building Deep Learning Pipelines Using Pytorch-Lightning
# -----------------------------------------
# Step 1: Define the Pytorch-Lightning Module with training process, validation process, and optimizer configuration.


class EEGClassifier(LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = F.cross_entropy(logits, y.long())
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = F.cross_entropy(logits, y.long())

        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.hparams.lr)

        return [optimizer], []


######################################################################
# Step 2: Initialize the Dataset
#
# We use the DEAP dataset supported by TorchEEG. Here, we set an EEG sample to 1 second long and include 128 data points. The baseline signal is 3 seconds long, cut into three, and averaged as the baseline signal for the trial. In offline preprocessing, we divide the EEG signal of every electrode into 4 sub-bands, and calculate the differential entropy on each sub-band as a feature, followed by debaselining and mapping on the grid. Finally, the preprocessed EEG signals are stored in the local IO. In online processing, all EEG signals are converted into Tensors for input into neural networks.
#

dataset = DEAPDataset(
    io_path=f'./tmp_out/examples_torch_lightning/deap',
    root_path='./tmp_in/data_preprocessed_python',
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
    ]))

######################################################################
# .. warning::
#    If you use TorchEEG under the `Windows` system and want to use multiple processes (such as in dataset or dataloader), you should check whether :obj:`__name__` is :obj:`__main__` to avoid errors caused by multiple :obj:`import`.
#
# That is, under the :obj:`Windows` system, you need to:
#  .. code-block::
#
#    if __name__ == "__main__":
#        dataset = DEAPDataset(
#                       io_path=f'./tmp_out/examples_torch_lightning/deap',
#                       root_path='./tmp_in/data_preprocessed_python',
#                       offline_transform=transforms.Compose([
#                           transforms.BandDifferentialEntropy(apply_to_baseline=True),
#                           transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
#                       ]),
#                       io_mode='pickle',
#                       online_transform=transforms.Compose(
#                           [transforms.BaselineRemoval(),
#                            transforms.ToTensor()]),
#                       label_transform=transforms.Compose([
#                           transforms.Select('valence'),
#                           transforms.Binary(5.0),
#                       ]))
#        # the following codes
#
# .. note::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 3: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using per-subject 5-fold cross-validation. In the process of division, we split the training and test sets separately on each subject's EEG samples. Here, we take 4 folds as training samples and 1 fold as test samples.
#

k_fold = KFold(n_splits=10,
               split_path='./tmp_out/examples_torch_lightning/split',
               shuffle=True,
               random_state=42)

######################################################################
# Step 4: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
#
# Next, we train the model for 50 epochs, with the Pytorch-Lightning module defined above wrapped in the :obj:`Trainer`. We use the :obj:`TensorBoardLogger` to record the training process and the :obj:`ModelCheckpoint` to save the model with the highest validation accuracy.
#

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    tb_logger = TensorBoardLogger(save_dir='lightning_logs',
                                  name=f'fold_{i + 1}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{epoch:02d}-{val_metric:.4f}",
        monitor='val_metric',
        mode='max')

    model = EEGClassifier(CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)))

    trainer = Trainer(max_epochs=50,
                      devices=2,
                      accelerator="auto",
                      strategy="ddp",
                      checkpoint_callback=checkpoint_callback,
                      logger=tb_logger)

    trainer.fit(model, train_loader, val_loader)
