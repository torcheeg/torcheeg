"""
Train a CCNN Model on the SEED Dataset
==========================

In this tutorial, we'll demonstrate how you can leverage TorchEEG to train a Continuous Convolutional Neural Network (CCNN) on the SEED dataset for emotion classification. Let's navigate through this process step-by-step, covering everything from dataset initialization to model evaluation.

"""

######################################################################
# Step 1: Initialize the Dataset
# ----------------------------------------------
#
# To begin with, we employ the SEED dataset provided by TorchEEG. Each EEG sample in this dataset lasts for 1 second and contains 200 data points.
# In terms of offline preprocessing, we perform the following tasks:
# - Divide each electrode's EEG signal into 4 frequency sub-bands
# - Compute the differential entropy of each sub-band to serve as a feature
# - Remove the baseline from the signal
# - Map the preprocessed signals onto a 2D grid
# For online processing, we convert the EEG signals into PyTorch Tensors, ensuring they are compatible with neural network inputs.
# 

from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LOCATION_DICT

dataset = SEEDDataset(io_path=f'./examples_seed_ccnn/seed',
                      root_path='./Preprocessed_EEG',
                      offline_transform=transforms.Compose([
                          transforms.BandDifferentialEntropy(sampling_rate=200),
                          transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
                      ]),
                      online_transform=transforms.ToTensor(),
                      label_transform=transforms.Compose([
                          transforms.Select('emotion'),
                          transforms.Lambda(lambda x: x + 1)
                      ]),
                      chunk_size=200,
                      num_worker=4)

######################################################################
# Step 2: Divide the Training and Test Samples in the Dataset
# ----------------------------------------------
#
# Next, let's partition our dataset into training and test sets using 5-fold cross-validation. We group the data based on their trial index, where each trial contributes 4 folds to the training set and 1 fold to the test set. These grouped samples are then combined to form the final training and test sets.
#

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=5, split_path=f'./examples_seed_ccnn/split')

######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
#
# We loop through each cross-validation set, and for each one, we
# initialize the CCNN model and define its hyperparameters. For instance,
# each EEG sample contains 4-channel features from 4 sub-bands, and the
# grid size is 9x9.
#
# We then train the model for 50 epochs using the ``ClassifierTrainer``.
#

from torch.utils.data import DataLoader
from torcheeg.models import CCNN

from torcheeg.trainers import ClassifierTrainer

import pytorch_lightning as pl

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=3, in_channels=4, grid_size=(9, 9))

    trainer = ClassifierTrainer(model=model,
                                num_classes=3,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_seed_ccnn/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')