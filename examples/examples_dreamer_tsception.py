"""
Train a Tsception Model on the DREAMER Dataset
==========================

In this tutorial, we'll showcase how you can use TorchEEG to train a Tsception model on the DREAMER dataset for the task of emotion classification. Let's go through the process step-by-step, covering everything from dataset preparation to model evaluation.

"""

######################################################################
# Step 1: Initialize the Dataset
# ----------------------------------------------
#
# First off, we'll use the DREAMER dataset provided by TorchEEG. Each EEG sample in this dataset spans 1 second and contains 128 data points. For each trial, the baseline signal lasts 61 seconds. 
# During offline preprocessing, we perform several steps:
# - Divide each electrode's EEG signal into 4 frequency sub-bands
# - Compute the differential entropy of each sub-band as a feature
# - Eliminate the baseline from the signal
# - Map the preprocessed signals onto a 2D grid
# For online processing, we convert the EEG signals into PyTorch Tensors to make them compatible with neural network inputs.
# Let's see how to accomplish these steps in code.
#

from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms

dataset = DREAMERDataset(io_path='./examples_dreamer_tsception/dreamer',
                         mat_path='./DREAMER.mat',
                         offline_transform=transforms.Compose([
                             transforms.BaselineRemoval(),
                             transforms.MeanStdNormalize(),
                             transforms.To2d()
                         ]),
                         online_transform=transforms.ToTensor(),
                         label_transform=transforms.Compose([
                             transforms.Select('valence'),
                             transforms.Binary(3.0)
                         ]),
                         chunk_size=128,
                         baseline_chunk_size=128,
                         num_baseline=61,
                         num_worker=4)

######################################################################
# Step 2: Divide the Training and Test Samples in the Dataset
# ----------------------------------------------
#
# Next, let's partition our dataset into training and test sets using 5-fold cross-validation. We group the data based on their trial index, where each trial contributes 4 folds to the training set and 1 fold to the test set. These grouped samples are then combined to form the final training and test sets.
#

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path=f'./examples_dreamer_tsception/split')

######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
#
# Having prepared and divided our dataset, we can now move on to model building and training. In each iteration of the cross-validation loop, we'll initialize the Tsception model and set its hyperparameters.
# Here, each EEG sample contains 128 data points across 14 electrodes. We'll train this model for 50 epochs using TorchEEG's `ClassifierTrainer`.
#

from torch.utils.data import DataLoader
from torcheeg.models import TSCeption

from torcheeg.trainers import ClassifierTrainer

import pytorch_lightning as pl

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = TSCeption(num_electrodes=14,
                      num_classes=2,
                      num_T=15,
                      num_S=15,
                      in_channels=1,
                      hid_channels=32,
                      sampling_rate=128,
                      dropout=0.5)

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_dreamer_tsception/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')