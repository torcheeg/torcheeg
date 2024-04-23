"""
Training Models Using Vanilla PyTorch
======================================

For those who prefer working directly with PyTorch's native functionalities instead of using third-party trainers, this tutorial provides a complete guide on how to train models using standard PyTorch features.

"""


######################################################################
# Pre-Experiment Preparation to Guarantee Reproducibility
# -------------------------------------------------------
# 
# It is recommended to utilize the logging module to record output in a
# log file, enabling easy referencing while simultaneously displaying it
# on the screen.
# 

import os
import time
import logging

os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./examples_vanilla_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)


######################################################################
# Setting the random number seed in all modules is essential to ensure
# consistent results upon repeated runs.
# 

import random
import numpy as np

import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


######################################################################
# Step 1: Initialize the Dataset
# ------------------------------
# 
# For this example, we utilize the DEAP dataset, as supported by TorchEEG.
# We configure an EEG sample to be 1 second long, encompassing 128 data
# points. A 3-second-long baseline signal is segmented into three parts
# and averaged to establish the baseline signal for the trial. During
# offline preprocessing, we divide each electrode’s EEG signal into 4
# sub-bands and compute the differential entropy on each sub-band as a
# feature, which is then subjected to debaselining and grid mapping. The
# preprocessed EEG signals are then stored in the local IO. In the case of
# online processing, all EEG signals are transformed into Tensors for
# input into the neural networks.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=f'./examples_vanilla_torch/deap',
    root_path='./tmp_in/data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.BaselineRemoval(),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
    ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)


######################################################################
# Step 2: Partition the Dataset into Training and Test Samples
# ------------------------------------------------------------
# 
# The dataset is divided using per-subject 5-fold cross-validation. During
# the division process, we separate the training and test sets based on
# each subject’s EEG samples. We allocate 4 folds to the training samples
# and 1 fold to the test samples.
# 

from torcheeg.model_selection import KFoldPerSubject

k_fold = KFoldPerSubject(n_splits=10,
                         split_path='./examples_vanilla_torch/split',
                         shuffle=True)


######################################################################
# The training process is customizable. You have the flexibility to define
# the training functions to facilitate the model’s training and testing.
# Here is a basic example:
# 

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# training process
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    record_step = int(len(dataloader) / 10)

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % record_step == 0:
            loss, current = loss.item(), batch_idx * len(X)
            logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


# validation process
def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    logger.info(
        f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n"
    )

    return correct, loss


######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
# 
# Initially, we use a loop to access the dataset in each cross-validation.
# During each cross-validation, we initialize the CCNN model and establish
# the hyperparameters. For instance, each EEG sample includes 4-channel
# features from 4 sub-bands, the grid size is set to 9x9, and so on.
# 
# Subsequently, we train the model for 50 epochs using the previously
# defined training function and monitor the model’s performance on the
# validation set at each epoch using the validation function defined
# earlier.
# 

import torch.nn as nn
from torcheeg.models import CCNN

from torcheeg.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

loss_fn = nn.CrossEntropyLoss()
batch_size = 64

test_accs = []
test_losses = []

for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)  # official: weight_decay=5e-1
    # split train and val
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        split_path=f'./examples_vanilla_torch/split{i}',
        shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    best_val_acc = 0.0
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(val_loader, model, loss_fn)
        # save the best model based on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./examples_vanilla_torch/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./examples_vanilla_torch/model{i}.pt'))
    test_acc, test_loss = valid(test_loader, model, loss_fn)

    # log the test result
    logger.info(
        f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}"
    )

    test_accs.append(test_acc)
    test_losses.append(test_loss)

# log the average test result on cross-validation datasets
logger.info(
    f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}"
)