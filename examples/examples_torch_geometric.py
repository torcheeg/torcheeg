import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torcheeg import transforms
from torcheeg.datasets import SEEDDataset
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_ADJACENCY_MATRIX
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNN(torch.nn.Module):
    def __init__(self, in_channels=4, num_layers=3, hid_channels=64, num_classes=3):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


if __name__ == "__main__":
    seed_everything(42)

    os.makedirs("./tmp_out/examples_torch_geometric", exist_ok=True)

    dataset = SEEDDataset(io_path=f'./tmp_out/examples_torch_geometric/seed',
                          root_path='./tmp_in/Preprocessed_EEG',
                          offline_transform=transforms.BandDifferentialEntropy(),
                          online_transform=transforms.pyg.ToG(SEED_ADJACENCY_MATRIX),
                          label_transform=transforms.Compose([
                              transforms.Select('emotion'),
                              transforms.Lambda(lambda x: int(x) + 1),
                          ]),
                          num_worker=8)

    k_fold = KFoldPerSubjectGroupbyTrial(n_splits=10, split_path=f'./tmp_out/examples_torch_geometric/split', shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):

        model = GNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            valid(val_loader, model, loss_fn)
        print("Done!")
