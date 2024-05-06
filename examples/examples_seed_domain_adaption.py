"""
Apply the Domain Adaption Algorithm on the SEED Dataset
===================================================

In this tutorial, you'll learn how to utilize TorchEEG's Associative Domain Adaptation (ADA) algorithm to address the cross-subject emotion recognition challenge using the SEED dataset.

"""

######################################################################
# Step 1: Initialize the Dataset
# ------------------------------
#
# We leverage the SEED dataset, supported by TorchEEG, where each EEG sample lasts 1 second and consists of 200 data points. 
#
# In offline preprocessing, each electrode's EEG signal is segmented into 4 sub-bands. We then calculate the differential entropy for each sub-band as a feature. The signals are debaselined, mapped onto a grid, and finally saved locally. 
#
# For online processing, we convert all EEG signals into Tensors for neural network input. Additionally, we apply after_hook_normalize to reduce trial variance effects on cross-subject emotion recognition.
#

from torcheeg.datasets import SEEDFeatureDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LOCATION_DICT
from torcheeg.transforms import after_hook_normalize
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

dataset = SEEDFeatureDataset(
    io_path=f'./examples_seed_domain_adaption/seed',
    root_path='./ExtractedFeatures',
    after_session=after_hook_normalize,
    offline_transform=transforms.Compose(
        [transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)]),
    online_transform=transforms.ToTensor(),  # seed do not have baseline signals
    label_transform=transforms.Compose(
        [transforms.Select('emotion'),
         transforms.Lambda(lambda x: x + 1)]),
    feature=['de_LDS'],
    num_worker=4)

######################################################################
# Step 2: Define the Model
# ----------------------------------------------
#
# We need to split the CCNN model into two parts: the feature extractor, and
# the classifier. The feature extractor is trained to extract domain-invariant
# features, while the classifier is trained to classify the emotion.
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


class Classifier(CCNN):
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


######################################################################
# Step 3: Split the Dataset and Train the Model
# -----------------------------------------------------
#
# In this case, we do not consider the cross-session problem. We use the
# subset of each session to train the model, and use the subset of the
# same session to test the model. We use the leave-one-subject-out method for evaluation. Each subject is iteratively left out from the training set and used as the test set.
#

from torcheeg.model_selection import Subcategory

subset = Subcategory(
    criteria='session_id',
    split_path=f'./examples_seed_domain_adaption/split/session')


######################################################################
# Step 4: Train the Model
# ----------------------------------------------
#
# For training, we make use of the ADA algorithm to adapt domain-invariant features across subjects.
#

from torch.utils.data import DataLoader
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.trainers import ADATrainer

scores = []
for j, sub_dataset in enumerate(subset.split(dataset)):
    loo = LeaveOneSubjectOut(
        split_path=f'./examples_seed_domain_adaption/split/loo_{j}')
    for i, (train_dataset, test_dataset) in enumerate(loo.split(sub_dataset)):

        extractor = Extractor(in_channels=5, num_classes=3)
        classifier = Classifier(in_channels=5, num_classes=3)

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

        trainer = ADATrainer(extractor=extractor,
                             classifier=classifier,
                             num_classes=2,
                             lr=1e-4,
                             weight_decay=0.0,
                             accelerator='gpu')

        trainer.fit(
            source_loader,
            target_loader,
            test_loader,
            max_epochs=50,
            default_root_dir=
            f'./examples_seed_domain_adaption/model/ses_{i}_sub_{j}.pth',
            callbacks=[ModelCheckpoint(save_last=True)],
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_val_batches=0.0)
        score = trainer.test(test_loader,
                             enable_progress_bar=True,
                             enable_model_summary=True)[0]
        scores.append(score['test_accuracy'])

    print(f'final accuracy: {np.array(scores).mean():>0.1f}%')