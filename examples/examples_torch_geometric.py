"""Training models with PyTorch Geometric
======================================
In this case, we introduce how to use TorchEEG and a customized training process based on vanilla PyTorch to train a PyTorch Geometric-based graph convolutional network on the SEED dataset for emotion classification.
"""

import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from torcheeg import transforms
from torcheeg.datasets import SEEDFeatureDataset
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_ADJACENCY_MATRIX
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
from torcheeg.transforms.pyg import ToG

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
# Defining a graph convolutional network
# -----------------------------------------
# Use the API provided by PyTorch Geometric to define graph convolutional networks. Here, the EEG signal or feature of the electrode corresponds to the input :obj`data.x`, and the relationship between electrodes corresponds to :obj`data.edge_index`. Depending on the definition of the adjacency matrix, the relationship may represent spatial adjacency, etc.
#


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels=5,
                 num_layers=3,
                 hid_channels=64,
                 num_classes=3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hid_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hid_channels, hid_channels))
        self.lin1 = Linear(hid_channels, hid_channels)
        self.lin2 = Linear(hid_channels, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


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

        if batch_idx % 20 == 0:
            loss, current = loss.item(), batch_idx * batch_size
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# validation process
def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n"
    )


######################################################################
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the SEED dataset supported by TorchEEG. Here we are using extracted EEG features. In the feature dataset, EEG signals (200 data points) per second are pre-computed with differential entropy in five sub-bands and smoothed using a linear dynamical system.
# .. note::
#   In online processing, All EEG signals are converted into a graph structure that can be processed by the PyTorch Geometric model, i.e., :obj:`torch_geometric.data.Data`, according to the adjacency matrix and the signals or features from electrodes. Here, electrodes represent nodes, and the adjacency matrix defines the association between electrodes.
#

dataset = SEEDFeatureDataset(io_path='./tmp_out/examples_torch_geometric/seed',
                             root_path='./tmp_in/ExtractedFeatures',
                             feature=['de_movingAve'],
                             online_transform=transforms.Compose([
                                 transforms.MinMaxNormalize(axis=-1),
                                 ToG(SEED_ADJACENCY_MATRIX)
                             ]),
                             label_transform=transforms.Compose([
                                 transforms.Select('emotion'),
                                 transforms.Lambda(lambda x: int(x) + 1),
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
#        dataset = SEEDFeatureDataset(io_path='./tmp_out/examples_torch_geometric/seed',
#                              root_path='./tmp_in/ExtractedFeatures',
#                              feature=['de_movingAve'],
#                              online_transform=transforms.Compose([
#                                  transforms.MinMaxNormalize(axis=-1),
#                                  ToG(SEED_ADJACENCY_MATRIX)
#                              ]),
#                              label_transform=transforms.Compose([
#                                  transforms.Select('emotion'),
#                                  transforms.Lambda(lambda x: int(x) + 1),
#                              ]),
#                              io_mode='pickle',
#                              num_worker=8)
#        # the following codes
#
# .. note::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using per-subject 5-fold cross-validation. In the process of division, we split the training and test sets separately on each subject's EEG samples. Here, we take 4 folds as training samples and 1 fold as test samples.
#

k_fold = KFoldPerSubjectGroupbyTrial(
    n_splits=10,
    split_path='./tmp_out/examples_torch_geometric/split',
    shuffle=False)

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the above-mentioned GNN model.
#
# Next, we train the model for 50 epochs using the training function defined above and report the model performance on the validation set at each epoch with the validation function defined above.
#

######################################################################
# .. note::
#    Please note that since the EEG signal sample returned by ToG is of type :obj:`torch_geometric.data.Data`, which represents a graph structure, the DataLoader provided by PyTorch cannot be used here to form a batch. Instead, :obj:`torch_geometric.loader.DataLoader` should be used to batch the adjacency matrix of the graph structure and the node features.
#

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
batch_size = 64

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = GNN().to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    epochs = 50
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        valid(val_loader, model, loss_fn)
