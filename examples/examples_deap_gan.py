"""GAN with the DEAP Dataset
======================================
In this case, we introduce how to use TorchEEG to train Generative Adversarial Networks (GAN) on the DEAP dataset for controllable EEG augmentation (which generates EEG signals confirming the given labels).

Here, a zero game is played to optimize a generator and a discriminator. The generator is used to generate EEG signal samples according to the given labels, and the discriminator is used to distinguish generated samples from real samples. By confusing the discriminator, the generator will be able to produce samples that approximate the real distribution.
"""

import logging
import os
import random
import time

import numpy as np
import torch
import torch.autograd as autograd
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LIST, DEAP_CHANNEL_LOCATION_DICT)
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import BCDiscriminator, BCGenerator
from torcheeg.trainers import CGANTrainer
from torcheeg.utils import plot_feature_topomap

###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.

os.makedirs('./tmp_out/examples_deap_gan/log', exist_ok=True)
logger = logging.getLogger('GAN with the DEAP Dataset')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/examples_deap_gan/log', f'{timeticks}.log'))
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
# TorchEEG provides many trainers to help complete the training of classification models, generative models and cross-domain methods. Here we choose the conditional generative adversarial network trainer (CGANTrainer), inherit the trainer and overload the log function to save the log using our own defined method; the :obj:`on_validation_step` hook function is overloaded to visualize EEG signals generated during validation using tensorboard and the tool function :obj:`plot_feature_topomap` in TorchEEG; other hook functions can also be overloaded to meet special needs.
#

writer = SummaryWriter(log_dir='./tmp_out/examples_deap_gan/vis',
                       comment='examples_deap_gan')


def gradient_penalty(model, real, fake, label=None):
    device = real.device
    real = real.data
    fake = fake.data
    alpha = torch.rand(real.size(0), *([1] * (len(real.shape) - 1))).to(device)
    inputs = alpha * real + ((1 - alpha) * fake)
    inputs.requires_grad_()

    if label is None:
        outputs = model(inputs)
    else:
        outputs = model(inputs, label)

    gradient = autograd.grad(outputs=outputs,
                             inputs=inputs,
                             grad_outputs=torch.ones_like(outputs).to(device),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

    gradient = gradient.flatten(1)
    return ((gradient.norm(2, dim=1) - 1)**2).mean()


class MyCGANTrainer(CGANTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)

    def before_validation_epoch(self, epoch_id, num_epochs, **kwargs):
        # record the current epoch
        self.cur_epoch = epoch_id
        self.val_g_loss.reset()
        self.val_d_loss.reset()

    def on_validation_step(self, val_batch, batch_id, num_batches, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        # for g_loss
        z = torch.normal(mean=0,
                         std=1,
                         size=(X.shape[0],
                               self.modules['generator'].in_channels)).to(
                                   self.device)
        gen_X = self.modules['generator'](z, y)
        g_loss = -torch.mean(self.modules['discriminator'](gen_X, y))

        # for d_loss
        real_loss = self.modules['discriminator'](X, y)
        fake_loss = self.modules['discriminator'](gen_X.detach(), y)
        gp_term = gradient_penalty(self.modules['discriminator'], X, gen_X, y)
        d_loss = -torch.mean(real_loss) + torch.mean(
            fake_loss) + self.lambd * gp_term

        self.val_g_loss.update(g_loss)
        self.val_d_loss.update(d_loss)

        vis_batch = num_batches // 20
        if batch_id % vis_batch == 0:
            t = transforms.ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT)
            # center should be 0.0
            signal = t.reverse(eeg=gen_X[y == 0][0].detach().cpu().numpy() -
                               0.5)['eeg']
            top_img = plot_feature_topomap(
                torch.tensor(signal),
                channel_list=DEAP_CHANNEL_LIST,
                feature_list=["theta", "alpha", "beta", "gamma"])
            # generate the visualization results and record them for the current epoch
            writer.add_image(f'top{batch_id}/eeg-0',
                             top_img,
                             self.cur_epoch,
                             dataformats='HWC')

            signal = t.reverse(eeg=gen_X[y == 1][0].detach().cpu().numpy() -
                               0.5)['eeg']
            top_img = plot_feature_topomap(
                torch.tensor(signal),
                channel_list=DEAP_CHANNEL_LIST,
                feature_list=["theta", "alpha", "beta", "gamma"])
            writer.add_image(f'top{batch_id}/eeg-1',
                             top_img,
                             self.cur_epoch,
                             dataformats='HWC')


######################################################################
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the DEAP dataset supported by TorchEEG. Here, we set an EEG sample to 1 second long and include 128 data points. The baseline signal is 3 seconds long, cut into three, and averaged as the baseline signal for the trial. In offline preprocessing, we divide the EEG signal of every electrode into 4 sub-bands, and calculate the differential entropy on each sub-band as a feature, followed by debaselining and mapping on the grid. Finally, the preprocessed EEG signals are stored in the local IO. In online processing, all EEG signals are converted into Tensors and normalized (in GANs normalization helps with convergence) for input into neural networks.
#

dataset = DEAPDataset(io_path=f'./tmp_out/examples_deap_gan/deap',
                      root_path='./tmp_in/data_preprocessed_python',
                      offline_transform=transforms.Compose([
                          transforms.BandDifferentialEntropy(
                              sampling_rate=128, apply_to_baseline=True),
                          transforms.BaselineRemoval(),
                          transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                      ]),
                      online_transform=transforms.Compose([
                          transforms.MinMaxNormalize(),
                          transforms.ToTensor(),
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select('valence'),
                          transforms.Binary(5.0),
                      ]),
                      chunk_size=128,
                      baseline_chunk_size=128,
                      num_baseline=3,
                      num_worker=4)

######################################################################
# .. warning::
#    If you use TorchEEG under the `Windows` system and want to use multiple processes (such as in dataset or dataloader), you should check whether :obj:`__name__` is :obj:`__main__` to avoid errors caused by multiple :obj:`import`.
#
# That is, under the :obj:`Windows` system, you need to:
#  .. code-block::
#
#    if __name__ == "__main__":
#        dataset = DEAPDataset(
#             io_path=
#             f'./tmp_out/examples_deap_gan/deap',
#             root_path='./tmp_in/data_preprocessed_python',
#             offline_transform=transforms.Compose([
#                 transforms.BandDifferentialEntropy(sampling_rate=128,
#                                                    apply_to_baseline=True),
#                 transforms.BaselineRemoval(),
#                 transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
#             ]),
#             online_transform=transforms.Compose([
#                 transforms.MinMaxNormalize(),
#                 transforms.ToTensor(),
#             ]),
#             label_transform=transforms.Compose([
#                 transforms.Select('valence'),
#                 transforms.Binary(5.0),
#             ]),
#             chunk_size=128,
#             baseline_chunk_size=128,
#             num_baseline=3,
#             num_worker=4)
#        # the following codes
#
# .. warning::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, we group according to the trial index, and every trial takes 4 folds as training samples and 1 fold as test samples. Samples across trials are aggregated to obtain training set and test set.
#

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='./tmp_out/examples_deap_gan/split')

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the generator and discriminator models and define the hyperparameters. For example, For example, we want to generate the differential entropy features of the 4 sub-bands of the simulated EEG signal.  The generated samples are sampled and transformed from a feature space of 128.
#
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # Initialize the model
    generator = BCGenerator(in_channels=128, out_channels=4, num_classes=2)
    discriminator = BCDiscriminator(hid_channels=128,
                                    in_channels=4,
                                    num_classes=2)

    # Initialize the trainer and use the 0-th GPU for training
    trainer = MyCGANTrainer(generator=generator,
                            discriminator=discriminator,
                            generator_lr=0.0001,
                            discriminator_lr=0.00001,
                            weight_decay=0,
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
    trainer.save_state_dict(f'./tmp_out/examples_deap_gan/weight/{i}.pth')
