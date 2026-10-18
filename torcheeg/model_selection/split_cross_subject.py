import os
from typing import Tuple, Callable, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path
from .utils import create_subset, ensure_split_cache, read_indices, write_indices

__all__ = ['train_test_split_cross_subject']


def train_test_split_cross_subject(
        dataset: BaseDataset,
        test_size: Union[int, float] = 0.2,
        shuffle: bool = False,
        label_transform: Callable = None,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        timeout: float = 300.0) -> Tuple[BaseDataset, BaseDataset]:
    r'''
    A tool function for cross-validations, to divide the training set and the test set across subjects. It is suitable for experiments with large dataset volume and no need to use k-fold cross-validations. A certain proportion of subjects are sampled as the test dataset, and samples from other subjects are used as training samples. In most literatures, 20% of the subjects are sampled for testing.

    :obj:`train_test_split_cross_subject` divides training set and the test set at the dimension of subjects. For example, when :obj:`test_size=0.2`, 80% of subjects are used for training, and 20% of subjects are used for testing. It is more consistent with real applications and can test the generalization of the model across different subjects.

    .. image:: _static/splits/train_test_split_cross_subject.svg
        :width: 100%
        :alt: Whole subjects are assigned to either training or test
        :align: center

    Training and test sets contain disjoint subject IDs. All samples from one
    subject remain in the same subset.

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.model_selection import train_test_split_cross_subject
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
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
        timeout (float): Lock wait in seconds; -1 waits indefinitely.
            (default: :obj:`300.0`)
    '''
    if split_path is None:
        split_path = get_random_dir_path(dir_prefix='model_selection')

    def write_split(split_path):
        info = dataset.info.reset_index(drop=True)
        subject_ids = list(pd.unique(info['subject_id']))

        if label_transform is not None:
            subject_labels = []
            for subject_id in subject_ids:
                subject_info = info[info['subject_id'] == subject_id]
                subject_label = subject_info.apply(
                    lambda info: label_transform(y=info.to_dict())['y'], axis=1).mean()
                subject_labels.append(subject_label)

            train_subject_ids, test_subject_ids = model_selection.train_test_split(
                subject_ids,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=subject_labels)
        else:
            train_subject_ids, test_subject_ids = model_selection.train_test_split(
                subject_ids,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle)

        if len(train_subject_ids) == 0 or len(test_subject_ids) == 0:
            raise ValueError(f'The number of training or testing subjects is zero.')

        train_info = info[info['subject_id'].isin(train_subject_ids)]
        test_info = info[info['subject_id'].isin(test_subject_ids)]

        write_indices(os.path.join(split_path, 'train.csv'),
                      train_info.index.to_numpy())
        write_indices(os.path.join(split_path, 'test.csv'), test_info.index.to_numpy())

    configuration = {
        'splitter': 'train_test_split_cross_subject',
        'test_size': test_size,
        'shuffle': shuffle,
        'random_state': random_state
    }
    if label_transform is not None:
        configuration['subject_labels'] = [
            (subject, dataset.info[dataset.info['subject_id'] == subject].apply(
                lambda row: label_transform(y=row.to_dict())['y'], axis=1).mean())
            for subject in pd.unique(dataset.info['subject_id'])
        ]
    ensure_split_cache(dataset.info, ['subject_id'],
                       split_path,
                       configuration, ['train.csv', 'test.csv'],
                       write_split,
                       timeout=timeout)

    train_info = read_indices(os.path.join(split_path, 'train.csv'), len(dataset))
    test_info = read_indices(os.path.join(split_path, 'test.csv'), len(dataset))

    train_dataset = create_subset(dataset, train_info)

    test_dataset = create_subset(dataset, test_info)

    return train_dataset, test_dataset
