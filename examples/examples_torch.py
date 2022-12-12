"""Training models with vanilla PyTorch
======================================
In this case, we introduce how to use TorchEEG and a customized training process based on vanilla PyTorch to train a Continuous Convolutional Neural Network (CCNN) on the DEAP dataset for emotion classification.
"""

import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldPerSubject, train_test_split
from torcheeg.models import CCNN

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/examples_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/examples_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

###############################################################################
# Customize the training process
# -----------------------------------------
# TorchEEG provides a large number of trainers to help complete the training of classification models, however, you can also define the training functions to complete the training and testing of the model. Here is a simple example:
#


# training process
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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

        if batch_idx % 100 == 0:
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
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the DEAP dataset supported by TorchEEG. Here, we set an EEG sample to 1 second long and include 128 data points. The baseline signal is 3 seconds long, cut into three, and averaged as the baseline signal for the trial. In offline preprocessing, we divide the EEG signal of every electrode into 4 sub-bands, and calculate the differential entropy on each sub-band as a feature, followed by debaselining and mapping on the grid. Finally, the preprocessed EEG signals are stored in the local IO. In online processing, all EEG signals are converted into Tensors for input into neural networks.
#

dataset = DEAPDataset(
    io_path=f'./tmp_out/examples_torch/deap',
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
# .. warning::
#    If you use TorchEEG under the `Windows` system and want to use multiple processes (such as in dataset or dataloader), you should check whether :obj:`__name__` is :obj:`__main__` to avoid errors caused by multiple :obj:`import`.
#
# That is, under the :obj:`Windows` system, you need to:
#  .. code-block::
#
#    if __name__ == "__main__":
#        dataset = DEAPDataset(io_path=f'./tmp_out/examples_ccnn/deap',
#                              root_path='./tmp_in/data_preprocessed_python',
#                              offline_transform=transforms.Compose([
#                                  transforms.BandDifferentialEntropy(apply_to_baseline=True),
#                                  transforms.BaselineRemoval(),
#                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
#                              ]),
#                              online_transform=transforms.ToTensor(),
#                              label_transform=transforms.Compose([
#                                  transforms.Select('valence'),
#                                  transforms.Binary(5.0),
#                              ]),
#                              num_worker=8)
#        # the following codes
#
# .. warning::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using per-subject 5-fold cross-validation. In the process of division, we split the training and test sets separately on each subject's EEG samples. Here, we take 4 folds as training samples and 1 fold as test samples.
#

k_fold = KFoldPerSubject(n_splits=10,
                         split_path='./tmp_out/examples_ccnn/split',
                         shuffle=True)

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
#
# Next, we train the model for 50 epochs using the training function defined above and report the model performance on the validation set at each epoch with the validation function defined above.
#

device = "cuda" if torch.cuda.is_available() else "cpu"
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
        split_path=f'./tmp_out/examples_ccnn/split{i}',
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
                       f'./tmp_out/examples_ccnn/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./tmp_out/examples_ccnn/model{i}.pt'))
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
