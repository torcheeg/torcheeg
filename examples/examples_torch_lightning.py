import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFold
from torchmetrics import Accuracy


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(4, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(256, 64, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Linear(9 * 9 * 64, 1024)
        self.lin2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class EEGClassifier(LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = F.cross_entropy(logits, y.long())
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[0]
        y = batch[1]

        logits = self.forward(X)
        loss = F.cross_entropy(logits, y.long())

        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        return [optimizer], []


if __name__ == "__main__":
    seed_everything(42)

    os.makedirs("./tmp_out/examples_torch_lightning", exist_ok=True)

    dataset = DEAPDataset(io_path=f'./tmp_out/examples_torch_lightning/deap',
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
                          ]))
    k_fold = KFold(n_splits=10,
                          split_path=f'./tmp_out/examples_torch_lightning/split',
                          shuffle=True,
                          random_state=42)

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        tb_logger = TensorBoardLogger(save_dir='lightning_logs', name=f'fold_{i + 1}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{val_metric:.4f}",
                                              monitor='val_metric',
                                              mode='max')

        model = EEGClassifier(CNN())

        trainer = Trainer(max_epochs=50,
                          devices=2,
                          accelerator="auto",
                          strategy="ddp",
                          checkpoint_callback=checkpoint_callback,
                          logger=tb_logger)

        trainer.fit(model, train_loader, val_loader)
