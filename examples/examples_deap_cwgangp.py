"""
Train a C-WGAN-GP Model on the DEAP Dataset
===================================================

In this tutorial, we'll walk through how to train a Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (C-WGAN-GP) on the DEAP EEG dataset. This will allow us to generate new EEG signals. We'll also evaluate the generated EEG signals using Fréchet Inception Distance (FID) and Kernel Inception Distance (KID).

"""


######################################################################
# Step 1: Initialize the Dataset
# ------------------------------
# 
# We use the DEAP dataset supported by TorchEEG. Each EEG sample is set to
# be 1 second long, encompassing 128 data points. The baseline signal is 3
# seconds long, which we divide into three sections and then average to
# obtain the trial’s baseline signal.
# 
# During offline preprocessing, we divide each electrode’s EEG signal into
# 4 sub-bands, calculate the differential entropy for each sub-band as a
# feature, perform debaselining, and map onto a grid. Finally, the
# preprocessed EEG signals are saved locally. For online processing, we
# convert all EEG signals into Tensors, making them suitable for neural
# network input.
# 

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=f'./examples_deap_cwgangp/deap',
    root_path='./data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)


######################################################################
# Step 2: Split the Dataset into Training and Test Sets
# -----------------------------------------------------
# 
# In this case, we use per-subject 5-fold cross-validation to split the
# dataset. During this process, we separate each subject’s EEG samples
# into training and test sets. We use 4 folds for training and 1 fold for
# testing.
# 

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='./examples_deap_cwgangp/split',
                           shuffle=True,
                           random_state=42)


######################################################################
# Step 3: Define the Model
# ----------------------------------------------
#
# We'll employ the CCNN model's feature extraction part to evaluate our CWGAN-GP model. Specifically, we'll compare the features from generated and real EEG signals.
#

from torcheeg.models import CCNN


class Extractor(CCNN):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        return x

######################################################################
# Step 3: Define the Model and Initiate Training
# ----------------------------------------------
# 
# Now we're ready to train the classifier and feature extractor, followed by the CWGAN-GP model.
# 

from torch.utils.data import DataLoader
from torcheeg.trainers import ClassifierTrainer, CWGANGPTrainer
from torcheeg.models import CCNN, BCGenerator, BCDiscriminator

import pytorch_lightning as pl

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    classifier = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    # if you have a pre-trained classifier, you can just load it, instead of training it from scratch
    trainer = ClassifierTrainer(model=classifier,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./examples_deap_cwgangp/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    
    extractor = Extractor(num_classes=2, in_channels=4, grid_size=(9, 9))
    extractor.load_state_dict(classifier.state_dict())

    g_model = BCGenerator(in_channels=128)
    d_model = BCDiscriminator(in_channels=4)

    trainer = CWGANGPTrainer(g_model,
                              d_model,
                              metric_extractor=extractor,
                              metric_classifier=classifier,
                              metric_num_features=9 * 9 * 64,
                              metrics=['fid', 'is'],
                             accelerator='gpu')
    trainer.fit(train_loader, val_loader, max_epochs=1)
    trainer.test(val_loader)

######################################################################
# 
# That's it! You've successfully trained a C-WGAN-GP model on the DEAP EEG dataset and evaluated it using FID and KID metrics.
#