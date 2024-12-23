import logging
import os
from copy import copy
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path

log = logging.getLogger('torcheeg')


def train_test_split_cross_subject(dataset: BaseDataset,
                                   test_size: float = 0.2,
                                   shuffle: bool = False,
                                   label_transform: Callable = None,
                                   random_state: Union[float, None] = None,
                                   split_path: Union[None, str] = None):
    r'''
    A tool function for cross-validations, to divide the training set and the test set across subjects. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. A certain proportion of subjects are sampled as the test dataset, and samples from other subjects are used as training samples. In most literatures, 20% of the subjects are sampled for testing.

    :obj:`train_test_split_cross_subject` divides training set and the test set at the dimension of subjects. For example, when :obj:`test_size=0.2`, 80% of subjects are used for training, and 20% of subjects are used for testing. It is more consistent with real applications and can test the generalization of the model across different subjects.

    .. image:: _static/train_test_split_cross_subject.png
        :alt: The schematic diagram of train_test_split_cross_subject
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_cross_subject
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))

        train_dataset, test_dataset = train_test_split_cross_subject(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (float):  Should be between 0.0 and 1.0 and represent the proportion of the subjects to include in the test split. (default: :obj:`0.2`)
        shuffle (bool): Whether to shuffle the subjects before splitting. (default: :obj:`False`)
        label_transform (Callable, optional): Function that returns the stratified label for each sample. If set to None, it will not be stratified. (default: :obj:`None`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the subjects, which controls the randomness of the split. Otherwise, this parameter has no effect. (default: :obj:`None`)
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
        subject_ids = list(set(info['subject_id']))

        if label_transform:
            subject_labels = []
            for subject_id in subject_ids:
                subject_info = info[info['subject_id'] == subject_id]
                subject_label = subject_info.apply(lambda info: label_transform(y=info.to_dict())['y'], axis=1).mean()
                subject_labels.append(subject_label)

            train_subject_ids, test_subject_ids = model_selection.train_test_split(
                subject_ids,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=subject_labels
            )
        else:
            train_subject_ids, test_subject_ids = model_selection.train_test_split(
                subject_ids,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle
            )

        if len(train_subject_ids) == 0 or len(test_subject_ids) == 0:
            raise ValueError(
                f'The number of training or testing subjects is zero.')

        train_info = info[info['subject_id'].isin(train_subject_ids)]
        test_info = info[info['subject_id'].isin(test_subject_ids)]

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
