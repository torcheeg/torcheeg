import logging
import os
from copy import copy
from typing import Union

import numpy as np
import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path

log = logging.getLogger('torcheeg')


def train_test_split_per_subject_cross_trial(dataset: BaseDataset,
                                             test_size: float = 0.2,
                                             subject: str = 's01.dat',
                                             shuffle: bool = False,
                                             random_state: Union[float,
                                                                 None] = None,
                                             split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for subject dependent experiments with large dataset volume and no need to use k-fold cross-validations. For the first step, the EEG signal samples of the specified user are selected. Then, parts of trials are sampled according to a certain proportion as the test dataset, and samples from other trials are used as training samples. In most literatures, 20% of the data are sampled for testing.

    .. image:: _static/train_test_split_per_subject_cross_trial.png
        :alt: The schematic diagram of train_test_split_per_subject_cross_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_per_subject_cross_trial
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_per_subject_cross_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        subject (str): The subject whose EEG samples will be used for training and test. (default: :obj:`s01.dat`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    if not os.path.exists(split_path):
        log.info(f'📊 | Create the split of train and test set.')
        log.info(
            f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
        )
        os.makedirs(split_path)
        info = dataset.info
        subjects = list(set(info['subject_id']))

        assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        subject_info = info[info['subject_id'] == subject]
        trial_ids = list(set(subject_info['trial_id']))

        train_trial_ids, test_trial_ids = model_selection.train_test_split(
            trial_ids,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state)

        if len(train_trial_ids) == 0 or len(test_trial_ids) == 0:
            raise ValueError(
                f'The number of training or testing trials for subject {subject} is zero.'
            )

        train_info = []
        for train_trial_id in train_trial_ids:
            train_info.append(
                subject_info[subject_info['trial_id'] == train_trial_id])
        train_info = pd.concat(train_info, ignore_index=True)

        test_info = []
        for test_trial_id in test_trial_ids:
            test_info.append(
                subject_info[subject_info['trial_id'] == test_trial_id])
        test_info = pd.concat(test_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, 'train.csv'), index=False)
        test_info.to_csv(os.path.join(split_path, 'test.csv'), index=False)

    else:
        log.info(
            f'📊 | Detected existing split of train and test set, use existing split from {split_path}.'
        )
        log.info(
            f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
        )

    train_info = pd.read_csv(os.path.join(split_path, 'train.csv'))
    test_info = pd.read_csv(os.path.join(split_path, 'test.csv'))

    train_dataset = copy(dataset)
    train_dataset.info = train_info

    test_dataset = copy(dataset)
    test_dataset.info = test_info

    return train_dataset, test_dataset