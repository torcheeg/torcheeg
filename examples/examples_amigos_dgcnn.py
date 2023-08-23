"""
Train a DGCNN Model on the AMIGOS Dataset
=============================

In this tutorial, we'll walk through how to leverage TorchEEG for training a Dynamical Graph Convolutional Neural Network (DGCNN) to classify emotions using the AMIGOS dataset. Let's walk through the essential steps—from dataset preparation to model evaluation.

"""

######################################################################
# Step 1: Initialize the Dataset
# ----------------------------------------------
#
# First, let's prepare the AMIGOS dataset using TorchEEG's built-in functionality. Each sample in the dataset is 1 second long, containing 128 EEG data points. We will use several preprocessing steps to prepare the data for our neural network model.
# Here's what each preprocessing step does:
# - Divide each EEG signal into 4 sub-bands
# - Calculate differential entropy for each sub-band as a feature
# - Remove the baseline from the signal
# - Convert the preprocessed signals into Tensors
# The code below demonstrates these steps:
# 

from torcheeg.datasets import AMIGOSDataset
from torcheeg import transforms

dataset = AMIGOSDataset(io_path=f'./examples_amigos_dgcnn/amigos',
                        root_path='./data_preprocessed',
                        offline_transform=transforms.BandDifferentialEntropy(
                            sampling_rate=128, apply_to_baseline=True),
                        online_transform=transforms.Compose([
                            transforms.BaselineRemoval(),
                            transforms.ToTensor()
                        ]),
                        label_transform=transforms.Compose([
                            transforms.Select('valence'),
                            transforms.Binary(5.0)
                        ]),
                        chunk_size=128,
                        baseline_chunk_size=128,
                        num_baseline=5,
                        num_worker=4)

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
# ----------------------------------------------
#
# Now, let's partition the dataset into training and test sets. We will employ 5-fold cross-validation for this purpose. During this phase, samples are grouped based on their trial index. Four folds are designated for training, and one fold serves as the test set. Subsequently, we combine these samples across all trials to finalize the training and test sets.
#

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path=f'./examples_amigos_dgcnn/split')

######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
#
# With our dataset prepared and divided, the next step is to build and train the DGCNN model. In this block of code, we define our DGCNN model's architecture, setting important hyperparameters like the number of channels, electrodes, and hidden layers.
# After that, we'll train the model for 50 epochs, utilizing TorchEEG's `ClassifierTrainer`.
#

from torch.utils.data import DataLoader
from torcheeg.models import DGCNN

from torcheeg.trainers import ClassifierTrainer

import pytorch_lightning as pl

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = DGCNN(in_channels=4,
                  num_electrodes=14,
                  hid_channels=64,
                  num_classes=2)

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_amigos_dgcnn/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')