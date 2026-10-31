import os
from typing import Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path
from .utils import create_subset, ensure_cache, read_indices, write_indices


def train_test_split_cross_trial(
        dataset: BaseDataset,
        test_size: Union[int, float] = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None) -> Tuple[BaseDataset, BaseDataset]:
    r'''
    A tool function for cross-validations, to divide the training set and the test set. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. Parts of trials are sampled according to a certain proportion as the test dataset, and samples from other trials are used as training samples. In most literatures, 20% of the data are sampled for testing.

    Trials are partitioned separately within each subject. With :obj:`shuffle=False` and :obj:`test_size=0.2`, the first 80% of each subject's trial IDs, in their order of appearance, are used for training and the remaining 20% for testing. All samples of a subject-trial pair stay together.

    .. image:: _static/splits/train_test_split_cross_trial.svg
        :width: 100%
        :alt: The schematic diagram of train_test_split_cross_trial
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_cross_trial
        from torcheeg import transforms
        from torch.utils.data import DataLoader

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

        train_dataset, test_dataset = train_test_split_cross_trial(dataset=dataset)

        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        ...

    Args:
        dataset (BaseDataset): Dataset to be divided.
        test_size (int):  If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. (default: :obj:`0.2`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    def write_split(split_path):
        info = dataset.info.reset_index(drop=True)
        subjects = list(pd.unique(info['subject_id']))

        train_info = None
        test_info = None

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            trial_ids = list(pd.unique(subject_info['trial_id']))

            train_index_trial_ids, test_index_trial_ids = model_selection.train_test_split(
                trial_ids,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state)

            if len(train_index_trial_ids) == 0 or len(test_index_trial_ids) == 0:
                raise ValueError(
                    f'The number of training or testing trials for subject {subject} is zero.'
                )

            train_trial_ids = list(train_index_trial_ids)
            test_trial_ids = list(test_index_trial_ids)

            subject_train_info = []
            for train_trial_id in train_trial_ids:
                subject_train_info.append(
                    subject_info[subject_info['trial_id'] == train_trial_id])
            subject_train_info = pd.concat(
                subject_train_info, ignore_index=False)

            subject_test_info = []
            for test_trial_id in test_trial_ids:
                subject_test_info.append(
                    subject_info[subject_info['trial_id'] == test_trial_id])
            subject_test_info = pd.concat(
                subject_test_info, ignore_index=False)

            if train_info is None and test_info is None:
                train_info = [subject_train_info]
                test_info = [subject_test_info]
            else:
                train_info.append(subject_train_info)
                test_info.append(subject_test_info)

        train_info = pd.concat(train_info, ignore_index=False)
        test_info = pd.concat(test_info, ignore_index=False)

        write_indices(os.path.join(split_path, 'train.csv'),
                      train_info.index.to_numpy())
        write_indices(os.path.join(split_path, 'test.csv'),
                      test_info.index.to_numpy())

    configuration = {
        'splitter': 'train_test_split_cross_trial',
        'test_size': test_size,
        'shuffle': shuffle,
        'random_state': random_state
    }
    ensure_cache(dataset.info, ['subject_id', 'trial_id'], split_path, configuration,
                 ['train.csv', 'test.csv'], write_split)

    train_info = read_indices(os.path.join(
        split_path, 'train.csv'), len(dataset))
    test_info = read_indices(os.path.join(
        split_path, 'test.csv'), len(dataset))

    train_dataset = create_subset(dataset, train_info)

    test_dataset = create_subset(dataset, test_info)

    return train_dataset, test_dataset
