import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import M3CVDataset
from torcheeg.models import TSCeption


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

        if batch_idx % 100 == 0:
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

    os.makedirs("./tmp_out/examples_bci_competition", exist_ok=True)

    train_dataset = M3CVDataset(
        io_path=f'./tmp_out/examples_bci_competition/m3cv_train',
        root_path='./tmp_in/aistudio',
        subset='Enrollment',
        channel_num=65,
        online_transform=transforms.Compose(
            [transforms.To2d(), transforms.ToTensor()]),
        label_transform=transforms.Compose(
            [transforms.Select('subject_id'),
             transforms.StringToInt()]),
        num_worker=4)

    val_dataset = M3CVDataset(
        io_path=f'./tmp_out/examples_bci_competition/m3cv_val',
        root_path='./tmp_in/aistudio',
        subset='Calibration',
        channel_num=65,
        online_transform=transforms.Compose(
            [transforms.To2d(), transforms.ToTensor()]),
        label_transform=transforms.Compose(
            [transforms.Select('subject_id'),
             transforms.StringToInt()]),
        num_worker=4)

    test_dataset = M3CVDataset(
        io_path=f'./tmp_out/examples_bci_competition/m3cv_test',
        root_path='./tmp_in/aistudio',
        subset='Testing',
        channel_num=65,
        online_transform=transforms.Compose(
            [transforms.To2d(), transforms.ToTensor()]),
        num_worker=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    model = TSCeption(num_electrodes=65, hid_channels=256,
                      num_classes=96).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    epochs = 50
    best_val_acc = 0.0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(val_loader, model, loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./tmp_out/examples_bci_competition/model.pt')

    model.load_state_dict(
        torch.load(f'./tmp_out/examples_bci_competition/model.pt'))

    preds = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for batch_idx, batch in enumerate(test_loader):
        X = batch[0].to(device)
        out = model(X)
        out = np.argmax(out.detach().cpu().numpy(), axis=-1)

        subject_ids = batch[1]['subject_id']
        epoch_ids = batch[1]['epoch_id']

        pred = []
        for i, epoch_id in enumerate(epoch_ids):
            if subject_ids[i] == 'None':
                out_subject = out[i]
            elif int(re.findall(r"\d+", subject_ids[i])[0]) == out[i]:
                out_subject = 1
            else:
                out_subject = 0
            pred.append({'EpochID': epoch_id, 'Prediction': out_subject})
        preds.extend(pred)

    preds_df = pd.DataFrame(preds)
    test_df = pd.read_csv('./tmp_in/aistudio/Testing_Info.csv')

    if len(preds_df) > len(test_df):
        preds_df = preds_df.groupby(['EpochID'])['Prediction'].agg(
            pd.Series.mode).reset_index(name='Prediction')

    result = pd.merge(test_df, preds_df, how='left', on=['EpochID'])
    result = result.drop(columns=['SubjectID', 'Session', 'Task', 'Usage'])

    result.to_csv('results.csv', index=False)
    print('Done!')