"""
Train a DARNet Model on the DTU Dataset
=============================

In this tutorial, we'll walk through how to leverage TorchEEG for training a DARNet Model to classify auditory attention using the DTU dataset. Let's walk through the essential steps—from dataset preparation to model evaluation.

"""

######################################################################
# Step 1: Initialize the Dataset
# ----------------------------------------------
#
# First, let's prepare the DTU dataset using TorchEEG's built-in functionality. Each sample in the dataset is 1 second long, containing 64 EEG data points.

import torch
import os
import numpy as np
import pytorch_lightning as pl
from mne.decoding import CSP
from torcheeg.trainers import ClassifierTrainer
from torcheeg.models import DARNet
from torch.utils.data import DataLoader
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
from torcheeg.datasets import DTUProcessedDataset
from torcheeg import transforms

dataset = DTUProcessedDataset(io_path=f'./.torcheeg/dtu',
                           root_path='./DATA_preproc',
                           label_transform=transforms.Compose([
                               transforms.Select('attended_speaker'),
                               transforms.Lambda(lambd=lambda x: x - 1)
                           ]),
                           chunk_size=64,
                           overlap=32,
                           num_worker=4)

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
# ----------------------------------------------
#
# Now, let's partition the dataset into training and test sets. We will employ 5-fold cross-validation for this purpose. During this phase, samples are grouped based on their trial index for each subject. Four folds are designated for training, and one fold serves as the test set.
#


k_fold = KFoldPerSubjectGroupbyTrial(
    n_splits=5, split_path=f'./torcheeg/dtu_kfold_per_subject')

######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
#
# With our dataset prepared and divided, the next step is to build and train the DARNet model. In this block of code, we define our DARNet model's architecture, setting important hyperparameters like the number of electrodes, sampling rate, and attention parameters.
# After that, we'll train the model for 50 epochs, utilizing TorchEEG's `ClassifierTrainer`.
#


metrics = []
current_subject_metrics = []

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    if i % 5 == 0 and i > 0:
        metrics.append(current_subject_metrics)
        mean_accuracy = np.mean(current_subject_metrics)
        current_subject_metrics = []

    train_dataset.online_transform = None
    val_dataset.online_transform = None

    csp = CSP(n_components=64, reg=None, log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    X = []
    y = []
    for sample in train_dataset:
        X.append(sample[0])
        y.append(sample[1])
    csp.fit(X, y)
    
    def csp_transform(eeg: np.ndarray) -> np.ndarray:
        eeg = np.expand_dims(eeg, axis=0)
        eeg = csp.transform(eeg)
        eeg = np.squeeze(eeg, axis=0).astype(np.float32)
        return eeg

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_loader.dataset.online_transform = transforms.Compose([
        transforms.Lambda(lambd=csp_transform),
        transforms.ToTensor(),
    ])
    val_loader.dataset.online_transform = transforms.Compose([
        transforms.Lambda(lambd=csp_transform),
        transforms.ToTensor(),
    ])

    model = DARNet(num_electrodes=64,
                   sampling_rate=64,
                   d_model=16,
                   num_heads=8,
                   attn_dropout=0.1,
                   num_classes=2)

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=5e-4,
                                weight_decay=3e-4,
                                accelerator="gpu")

    # https://github.com/fchest/DARNet/blob/6254e7ebd2aa2a63b64b407a1be3013088381c10/main.py
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f'./.torcheeg/checkpoints/dtu_kfold_per_subject_darnet/{i}',
            filename='best',
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min',
            save_last=True),
        pl.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, verbose=False, mode="min")
    ]

    trainer.fit(train_loader,
                val_loader,
                max_epochs=100,
                callbacks=callbacks,
                enable_progress_bar=True,
                enable_model_summary=True,
                )

    checkpoint = torch.load(os.path.join(
        f'./.torcheeg/checkpoints/dtu_kfold_per_subject_darnet/{i}', 'best.ckpt'))
    trainer.load_state_dict(checkpoint['state_dict'])

    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]

    test_accuracy = score['test_accuracy']
    current_subject_metrics.append(test_accuracy)

metrics.append(current_subject_metrics)
mean_accuracy = np.mean(current_subject_metrics)
print(
    f"\nSubject {len(metrics)} completed - Mean Accuracy: {mean_accuracy:.4f}\n")

######################################################################
# Step 4: Report Results
# ----------------------------------------------
#
# Calculate and report the average accuracy for each subject and the overall average accuracy across all subjects.
#

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}\n")

subject_mean_accuracies = []
for subject_idx, subject_metrics in enumerate(metrics):
    mean_accuracy = np.mean(subject_metrics)
    std_accuracy = np.std(subject_metrics)
    subject_mean_accuracies.append(mean_accuracy)
    print(
        f"Subject {subject_idx + 1}: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

overall_mean_accuracy = np.mean(subject_mean_accuracies)
overall_std_accuracy = np.std(subject_mean_accuracies)

print(f"\n{'='*70}")
print(
    f"Overall Mean Accuracy: {overall_mean_accuracy:.4f} ± {overall_std_accuracy:.4f}")
print(f"{'='*70}\n")
