"""Examples of MNEDataset
======================================
In this case, we will introduce how to combine TorchEEG with MNE, and use deep learning algorithms to analyze the existing :obj:`mne.Epochs` format data.
"""

import logging
import os
import random
import time

import mne
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MNEDataset
from torcheeg.model_selection import KFold
from torcheeg.models import TSCeption
from torcheeg.trainers import ClassificationTrainer

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/examples_mne_dataset/log', exist_ok=True)
logger = logging.getLogger('Examples of MNEDataset')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/examples_mne_dataset/log', f'{timeticks}.log'))
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
# Customize Trainer
# -----------------------------------------
# TorchEEG provides a large number of trainers to help complete the training of classification models, generative models and cross-domain methods. Here we choose the simplest classification trainer, inherit the trainer and overload the log function to save the log using our own defined method; other hook functions can also be overloaded to meet special needs.
#


class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


###############################################################################
# Read data using MNE and formalize as :obj:`mne.Epochs`
# -----------------------------------------
# We use mne's API to automatically download the motor imagery dataset in Physionet. The EEG signals of subjects 1-21 in runs 6, 10, and 14 were downloaded and filtered. We store multiple :obj:`mne.Epochs` into an array, and use a counterpart array :obj:`metadata_list` to describe the metadata corresponding to the corresponding Epochs.
#
metadata_list = [{
    'subject': subject_id,
    'run': run_id
} for subject_id in range(1, 22)
                 for run_id in [6, 10, 14]]  # motor imagery: hands vs feet

epochs_list = []
for metadata in metadata_list:
    physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                   metadata['run'],
                                                   update_path=False)[0]
    raw = mne.io.read_raw_edf(physionet_path, preload=True, stim_channel='auto')
    mne.datasets.eegbci.standardize(raw)

    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = mne.pick_types(raw.info,
                           meg=False,
                           eeg=True,
                           stim=False,
                           eog=False,
                           exclude='bads')
    # init Epochs with raw EEG signals and corresponding event annotations. Here, tmin is set to -1., and tmax is set to 4.0, to avoid classification of evoked responses by using epochs that start 1s after cue onset.
    epochs_list.append(
        mne.Epochs(raw,
                   events,
                   dict(hands=2, feet=3),
                   tmin=-1.,
                   tmax=4.0,
                   proj=True,
                   picks=picks))

###############################################################################
# Convert :obj:`mne.Epochs` into MNEDataset
# -----------------------------------------
# We use MNEDataset to window the Epochs. Here, we set the window size to 160 (1-second long) and the overlap to 80 to segment the EEG signal corresponding to each event. The corresponding information in :obj:`metadata_list` will be assigned to the corresponding window. At the same time, the window also includes the start position of the window :obj:`start_at`, the end position :obj:`start_at`, the epoch index :obj:`trial_id` and the corresponding event type :obj:`event`, which can be used and transformed as label.
#
dataset = MNEDataset(epochs_list=epochs_list,
                     metadata_list=metadata_list,
                     chunk_size=160,
                     overlap=80,
                     io_path='./tmp_out/examples_mne_dataset/physionet',
                     offline_transform=transforms.Compose(
                         [transforms.MeanStdNormalize(),
                          transforms.To2d()]),
                     online_transform=transforms.ToTensor(),
                     label_transform=transforms.Compose([
                         transforms.Select('event'),
                         transforms.Lambda(lambda x: x - 2)
                     ]),
                     num_worker=2)

######################################################################
# .. warning::
#    If you use TorchEEG under the `Windows` system and want to use multiple processes (such as in dataset or dataloader), you should check whether :obj:`__name__` is :obj:`__main__` to avoid errors caused by multiple :obj:`import`.
#
# That is, under the :obj:`Windows` system, you need to:
#  .. code-block::
#
#    if __name__ == "__main__":
#        dataset = MNEDataset(epochs_list=epochs_list,
#                      metadata_list=metadata_list,
#                      chunk_size=160,
#                      overlap=80,
#                      io_path='./tmp_out/examples_mne_dataset/physionet',
#                      offline_transform=transforms.Compose(
#                          [transforms.MeanStdNormalize(),
#                           transforms.To2d()]),
#                      online_transform=transforms.ToTensor(),
#                      label_transform=transforms.Compose([
#                          transforms.Select('event'),
#                          transforms.Lambda(lambda x: x - 2)
#                      ]),
#                      io_mode='pickle',
#                      num_worker=2)
#        # the following codes
#
# .. note::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, the total dataset takes 4 folds as training samples and 1 fold as test samples.
#
k_fold = KFold(n_splits=5, split_path='./tmp_out/examples_mne_dataset/split')

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation.
# In each cross-validation, we initialize the TSCeption model, and define the hyperparameters. For example, the sampling rate of EEG sample is 160, there are 15 temporal modules and 15 spatial modules, etc. In this example, the shape of EEG signals is :obj:`[60, 160]` and the number of classes is 2.
#
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.
#

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # Initialize the model
    model = TSCeption(num_electrodes=64,
                      num_classes=2,
                      num_T=15,
                      num_S=15,
                      in_channels=1,
                      hid_channels=32,
                      sampling_rate=160,
                      dropout=0.5)

    # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
    trainer = MyClassificationTrainer(model=model,
                                      lr=1e-4,
                                      weight_decay=1e-4,
                                      device_ids=[0])

    # Initialize several batches of training samples and test samples
    train_loader = DataLoader(train_dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=4)

    # Do 50 rounds of training
    trainer.fit(train_loader, val_loader, num_epochs=50)
    trainer.test(val_loader)
    trainer.save_state_dict(f'./tmp_out/examples_mne_dataset/weight/{i}.pth')
