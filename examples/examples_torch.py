import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFold


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN(torch.nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(in_channels, 64, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(256, 64, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Linear(9 * 9 * 64, 1024)
        self.lin2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
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

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
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

    os.makedirs("./tmp_out/examples_torch", exist_ok=True)

    dataset = DEAPDataset(io_path=f'./tmp_out/examples_torch/deap',
                          root_path='./tmp_in/data_preprocessed_python',
                          offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(apply_to_baseline=True),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
                          ]),
                          online_transform=transforms.Compose([transforms.BaselineRemoval(),
                                                               transforms.ToTensor()]),
                          label_transform=transforms.Compose([
                              transforms.Select('valence'),
                              transforms.Binary(5.0),
                          ]),
                          num_worker=4)

    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_torch/split', shuffle=True, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):

        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            valid(val_loader, model, loss_fn)
        print("Done!")
