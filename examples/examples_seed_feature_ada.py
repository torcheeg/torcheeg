"""Cross-domain Emotion Recognition with ADA
======================================
In this case, we introduce how to use TorchEEG and a simple CCNN model to complete emotion recognition across subjects. Here, the EEG signals of different subjects have a distribution gap, so the model trained on some subjects has a performance drop on unknown subjects. We use a cross-domain algorithm, Associative Domain Adaptation (ADA), to solve this problem.
"""

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader

from torcheeg import transforms
from torcheeg.datasets import SEEDFeatureDataset
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut, Subcategory
from torcheeg.models import CCNN
from torcheeg.trainers import ADATrainer

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/examples_seed_feature_ada/log', exist_ok=True)
logger = logging.getLogger('Cross-domain Emotion Recognition with DAD')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/examples_seed_feature_ada/log', f'{timeticks}.log'))
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
# Define feature extractor and classifier
# -----------------------------------------
# Different from the vanilla classification model, cross-domain algorithms usually need to constrain the features extracted by the model (such as domain invariant characteristics, etc.), so we need to split the classification model into feature extraction part and classification part.
#


class Extractor(CCNN):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        return x


class Classifier(CCNN):
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


###############################################################################
# Customize Trainer
# -----------------------------------------
# TorchEEG provides a large number of domain adaption trainers to help complete the training of cross-subject/session emotion classification models. Here we choose the associative domain adaptation trainer, inherit the trainer and overload the log function to save the log using our own defined method; other hook functions can also be overloaded to meet special needs.
#


class MyClassificationTrainer(ADATrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


######################################################################
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the SEED dataset supported by TorchEEG. Here we are using extracted EEG features. In the feature dataset, EEG signals (200 data points) per second are pre-computed with differential entropy in five sub-bands and smoothed using a linear dynamical system. We then normalize it and map it onto the grid. Finally, EEG signals are converted into Tensors for input into neural networks.
#

dataset = SEEDFeatureDataset(io_path='./tmp_out/examples_seed_feature_ada/seed',
                             root_path='./tmp_in/ExtractedFeatures',
                             feature=['de_movingAve'],
                             online_transform=transforms.Compose([
                                 transforms.MinMaxNormalize(axis=-1),
                                 transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                                 transforms.ToTensor()
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
#        dataset = SEEDFeatureDataset(io_path='./tmp_out/examples_seed_feature_ada/seed',
#                              root_path='./tmp_in/ExtractedFeatures',
#                              feature=['de_movingAve'],
#                              online_transform=transforms.Compose([
#                                  transforms.MinMaxNormalize(axis=-1),
#                                  transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
#                                  transforms.ToTensor()
#                              ]),
#                              label_transform=transforms.Compose([
#                                  transforms.Select('emotion'),
#                                  transforms.Lambda(lambda x: int(x) + 1),
#                              ]),
#                              num_worker=8)
#        # the following codes

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here we do not consider the impact of cross-session on the test results. Therefore, we first mark the session index on the sample according to the collection date. Next, we use :obj:`Subcategory` to divide the data set to obtain the sub-data set of the first session, the second session and the third session.
#

subject_info_list = []
for subject_id in dataset.info['subject_id'].unique().tolist():
    subject_info = dataset.info[dataset.info['subject_id'] == subject_id]
    session_id_set = subject_info['date'].unique().tolist()
    session_id_map = {
        session_id: i
        for i, session_id in enumerate(session_id_set)
    }
    subject_info['session_id'] = subject_info['date'].apply(
        lambda x: session_id_map[x])
    subject_info_list.append(subject_info)
dataset.info = pd.concat(subject_info_list)

subset = Subcategory(criteria='session_id',
                     split_path='./tmp_out/examples_seed_feature_ada/split')

######################################################################
# Step 3: Define the Model and Start Training
#
# For the dataset of each session, we use the leave-one-out method to conduct experiments, and use one subject as the test dataset and the other subjects as the training dataset to train the model.
#
# We define the training set as the source domain and the test set as the target domain, hoping that the model trained on the source domain can be generalized to the target domain. We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the source domain data samples and the target domain samples (without labels, as they are from test dataset) and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

for j, sub_dataset in enumerate(subset.split(dataset)):
    k_fold = LeaveOneSubjectOut(
        split_path=f'./tmp_out/examples_seed_feature_ada/split_{j}')
    for i, (train_dataset,
            test_dataset) in enumerate(k_fold.split(sub_dataset)):
        extractor = Extractor(in_channels=5, num_classes=3)
        classifier = Classifier(in_channels=5, num_classes=3)
        trainer = MyClassificationTrainer(extractor=extractor,
                                          classifier=classifier,
                                          lr=1e-4,
                                          lambd=1.0,
                                          weight_decay=0.0,
                                          device_ids=[0])

        source_loader = DataLoader(train_dataset,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=4,
                                   drop_last=True)

        target_loader = DataLoader(test_dataset,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=4,
                                   drop_last=True)

        test_loader = DataLoader(test_dataset,
                                 batch_size=128,
                                 shuffle=True,
                                 num_workers=4)
        trainer.fit(source_loader, target_loader, test_loader, num_epochs=50)
        trainer.test(test_loader)
        trainer.save_state_dict(
            f'./tmp_out/examples_seed_feature_ada/weight/{j}-{i}.pth')
