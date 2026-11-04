"""
Using TorchEEG to Complete a Deep Learning Workflow
===================================================

In this tutorial, we demonstrate a complete deep learning workflow using TorchEEG. We will cover the following aspects:

1. Utilizing Datasets and Transformers in TorchEEG
2. Data Partitioning Strategies in TorchEEG
3. Leveraging Models and Trainers in TorchEEG

"""


######################################################################
# Step 1: Initialize the Dataset
# ------------------------------
#
# We use the DEAP dataset supported by TorchEEG. Each EEG sample is set to
# be 1 second long, encompassing 128 data points. The baseline signal is 3
# seconds long, which we divide into three sections and then average to
# obtain the trial’s baseline signal.
#
# The offline function computes and caches band features. The online function
# removes the baseline and prepares EEG tensors on each read. label_fn
# constructs labels independently from the sample metadata.
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
    io_path=f'./examples_pipeline/deap',
    root_path='./data_preprocessed_python',
    num_worker=8,
    offline_transform=preprocess_fn,
    online_transform=transform_fn,
    label_transform=label_fn,
)


######################################################################
# Step 2: Split the Dataset into Training and Test Sets
# -----------------------------------------------------
#
# In this case, we use per-subject 5-fold cross-validation to split the
# dataset. During this process, we separate each subject’s EEG samples
# into training and test sets. We use 4 folds for training and 1 fold for
# testing.
#

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=10,
                           split_path='./examples_pipeline/split',
                           shuffle=True,
                           random_state=42)


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

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torcheeg.models import CCNN
from torcheeg.trainers import ClassifierTrainer

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_pipeline/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')
