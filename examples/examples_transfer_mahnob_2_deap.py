"""
Transfer Learning from MAHNOB-HCI to DEAP
==========================

In this tutorial, we demonstrate how to use TorchEEG to implement transfer learning with a Continuous Convolutional Neural Network (CCNN) model. We'll train our model on the MAHNOB-HCI dataset and then apply it to the DEAP dataset.

"""

######################################################################
# Step 1: Initialize the Dataset
# ------------------------------
#
# We use the MAHNOB dataset supported by TorchEEG. Each EEG sample is set to
# be 1 second long, encompassing 128 data points. The baseline signal is 30
# seconds long, which we divide into three sections and then average to
# obtain the trial’s baseline signal.
#
# During offline preprocessing, we divide each electrode’s EEG signal into
# 4 sub-bands, calculate the differential entropy for each sub-band as a
# feature, perform debaselining, and map onto a grid. Finally, the
# preprocessed EEG signals are saved locally. For online processing, we
# convert all EEG signals into Tensors, making them suitable for neural
# network input.
#
# We use the `after_hook_normalize` function to normalize each EEG trial. This helps in reducing the variance between trials, thereby aiding the transfer generalizable knowledge from MAHNOB to DEAP.
#

from torcheeg.datasets import MAHNOBDataset
from torcheeg import transforms
from torcheeg.datasets.constants import MAHNOB_CHANNEL_LOCATION_DICT

from torcheeg.transforms import after_hook_normalize

dataset = MAHNOBDataset(
    io_path='./examples_transfer_mahnob_2_deap/mahnob',
    root_path='./Sessions',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(sampling_rate=128,
                                           apply_to_baseline=True),
        transforms.BaselineRemoval(),
        transforms.ToGrid(MAHNOB_CHANNEL_LOCATION_DICT)
    ]),
    online_transform=transforms.ToTensor(),
    after_subject=after_hook_normalize,
    label_transform=transforms.Compose(
        [transforms.Select('feltVlnc'),
         transforms.Binary(5.0)]),
    chunk_size=128,
    baseline_chunk_size=128,
    num_baseline=30,
    num_worker=4)

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
# ===========================================================
#
# For this example, we'll use a simple train-test split to partition the MAHNOB-HCI dataset.
#

from torcheeg.model_selection import train_test_split_groupby_trial

train_dataset, val_dataset = train_test_split_groupby_trial(
    dataset,
    test_size=0.2,
    split_path=f'./examples_transfer_mahnob_2_deap/split/mahnob',
    shuffle=True)

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
# Here, we observe that different datasets have different level of class
# imbalance. To address this issue, we use the ImbalancedDatasetSampler
# from torchsampler to sample the training data.
#

import pytorch_lightning as pl

from torcheeg.models import CCNN
from torcheeg.trainers import ClassifierTrainer

from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          sampler=ImbalancedDatasetSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = CCNN(num_classes=3, in_channels=4, grid_size=(9, 9))

trainer = ClassifierTrainer(model=model,
                            num_classes=2,
                            lr=1e-4,
                            weight_decay=1e-4,
                            accelerator="gpu")

trainer.fit(
    train_loader,
    val_loader,
    max_epochs=50,
    default_root_dir=f'./examples_transfer_mahnob_2_deap/pretrained_model',
    callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
    enable_progress_bar=True,
    enable_model_summary=True,
    limit_val_batches=0.0)

state_dict = model.state_dict()

######################################################################
# Step 4: Initialize the Dataset
# ------------------------------
#
# We then switch to the DEAP dataset, which is also supported by TorchEEG. Similar to the MAHNOB-HCI dataset, each EEG sample is set to
# be 1 second long, encompassing 128 data points. The baseline signal is 3
# seconds long, which we divide into three sections and then average to
# obtain the trial’s baseline signal.
#
# During offline preprocessing, we divide each electrode’s EEG signal into
# 4 sub-bands, calculate the differential entropy for each sub-band as a
# feature, perform debaselining, and map onto a grid. Finally, the
# preprocessed EEG signals are saved locally. For online processing, we
# convert all EEG signals into Tensors, making them suitable for neural
# network input.
#

from torcheeg.datasets import DEAPDataset

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=f'./examples_transfer_mahnob_2_deap/deap',
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
# Step 5: Split the Dataset into Training and Test Sets
# -----------------------------------------------------
#
# In this case, we use 5-fold cross-validation to split the
# dataset. During this process, we separate EEG samples of each trial
# into training and test sets. We use 4 folds for training and 1 fold for
# testing.
#

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(
    n_splits=10,
    split_path='./examples_transfer_mahnob_2_deap/split/deap',
    shuffle=True,
    random_state=42)

######################################################################
# Step 6: Transfer Learning on the DEAP Dataset
# ----------------------------------------------
#
# Finally, we load the pre-trained CCNN model and fine-tune it on the DEAP dataset. We maintain the same set of hyperparameters for consistency.
#

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
    model.load_state_dict(state_dict)

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(
        train_loader,
        val_loader,
        max_epochs=50,
        default_root_dir=f'./examples_transfer_mahnob_2_deap/pretrained_model',
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_val_batches=0.0)

    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')