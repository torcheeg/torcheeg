import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BCI2022Dataset
from torcheeg.models import CCNN
from torcheeg.datasets.constants.emotion_recognition.bci2022 import \
    BCI2022_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut, train_test_split

PARAMS = {'lr': 0.0001, 'dropout': 0.5, 'hid_channels': 64, 'out_channels': 512}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        if batch_idx % 500 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


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
    print(
        f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n"
    )

    return correct, loss


if __name__ == "__main__":
    seed_everything(42)

    os.makedirs(f"./tmp_out/examples_bci2022_competition", exist_ok=True)

    dataset = BCI2022Dataset(
        io_path=f'./tmp_out/bci2022_train',
        root_path='./tmp_in/TrainSet',
        offline_transform=transforms.Concatenate([
            transforms.BandDifferentialEntropy(),
            transforms.BandPowerSpectralDensity()
        ]),
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=0),
            transforms.ToGrid(BCI2022_CHANNEL_LOCATION_DICT),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Select('emotion'),
        num_worker=8)

    k_fold = LeaveOneSubjectOut(split_path=f'./tmp_out/examples_bci2022_competition/split')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 100

    test_accs = []
    test_losses = []

    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):

        model = CCNN(num_classes=9, in_channels=8, grid_size=(8, 9)).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1e-4)  # official: weight_decay=5e-1

        train_dataset, val_dataset = train_test_split(
            train_dataset,
            test_size=0.2,
            split_path=f'./tmp_out/examples_bci2022_competition/split{i}',
            shuffle=True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True)

        epochs = 50
        best_val_acc = 0.0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=10,
                                                               eta_min=1e-5)

        for t in range(epochs):
            train_loss = train(train_loader, model, loss_fn, optimizer)
            val_acc, val_loss = valid(val_loader, model, loss_fn)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f'./tmp_out/examples_bci2022_competition/model{i}.pt')

            scheduler.step()

        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        model.load_state_dict(
            torch.load(f'./tmp_out/examples_bci2022_competition/model{i}.pt'))
        test_acc, test_loss = valid(test_loader, model, loss_fn)

        print(
            f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}"
        )

        test_accs.append(test_acc)
        test_losses.append(test_loss)

    print(
        f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}"
    )
