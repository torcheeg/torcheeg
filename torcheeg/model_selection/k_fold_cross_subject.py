import os
import re
from typing import Iterator, Callable, Dict, Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import write_indices

__all__ = ['KFoldCrossSubject']


class KFoldCrossSubject(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. One of the most commonly used data partitioning methods, where the data set is divided into k subsets of subjects, with one subset subjects being retained as the test set and the remaining k-1 subset subjects being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/splits/KFoldCrossSubject.svg
        :width: 100%
        :alt: The schematic diagram of KFoldCrossSubject
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldCrossSubject
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        cv = KFoldCrossSubject(n_splits=5, shuffle=True)
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

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that samples within each split will not be shuffled. (default: :obj:`False`)
        label_transform (Callable, optional): Function that returns the stratified label for each sample. If set to None, it will not be stratified. (default: :obj:`None`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. If shuffle is :obj:`False`, this parameter has no effect. (default: :obj:`None`)
        split_path (str): Path to data partition information. If the path exists, the existing partition will be read from it. If the path does not exist, the current division method will be saved for future use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 label_transform: Callable = None,
                 random_state: Union[None, int] = None,
                 split_path: Union[None, str] = None) -> None:

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(split_path=split_path)
        self.label_transform = label_transform

        if label_transform is not None:
            self._k_fold = model_selection.StratifiedKFold(n_splits=n_splits,
                                                           shuffle=shuffle,
                                                           random_state=random_state)
        else:
            self._k_fold = model_selection.KFold(n_splits=n_splits,
                                                 shuffle=shuffle,
                                                 random_state=random_state)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        subject_ids = list(pd.unique(info['subject_id']))
        if self.label_transform is not None:
            subject_labels = []
            for subject_id in subject_ids:
                subject_info = info[info['subject_id'] == subject_id]
                subject_label = subject_info.apply(
                    lambda info: self.label_transform(y=info.to_dict())['y'],
                    axis=1).mean()
                subject_labels.append(subject_label)
            enumerater = enumerate(self._k_fold.split(subject_ids, subject_labels))
        else:
            enumerater = enumerate(self._k_fold.split(subject_ids))
        for fold_id, (train_index_subject_ids, test_index_subject_ids) in enumerater:
            if len(train_index_subject_ids) == 0 or len(test_index_subject_ids) == 0:
                raise ValueError(f'The number of training or testing subjects is zero.')
            train_subject_ids = [
                subject_ids[index] for index in train_index_subject_ids
            ]
            test_subject_ids = [subject_ids[index] for index in test_index_subject_ids]
            train_info = []
            for train_subject_id in train_subject_ids:
                train_info.append(info[info['subject_id'] == train_subject_id])
            train_info = pd.concat(train_info, ignore_index=False)
            test_info = []
            for test_subject_id in test_subject_ids:
                test_info.append(info[info['subject_id'] == test_subject_id])
            test_info = pd.concat(test_info, ignore_index=False)
            write_indices(os.path.join(split_path, f'train_fold_{fold_id}.csv'),
                          train_info.index.to_numpy())
            write_indices(os.path.join(split_path, f'test_fold_{fold_id}.csv'),
                          test_info.index.to_numpy())

    @property
    def fold_ids(self):
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self.check(dataset, ['subject_id'])
        configuration = {}
        if self.label_transform is not None:
            configuration['subject_labels'] = [
                (subject, dataset.info[dataset.info['subject_id'] == subject].apply(
                    lambda row: self.label_transform(y=row.to_dict())['y'],
                    axis=1).mean()) for subject in pd.unique(dataset.info['subject_id'])
            ]
        self._prepare(dataset, ['subject_id'], configuration=configuration)

        fold_ids = self.fold_ids

        for fold_id in fold_ids:

            train_dataset = self._read_subset(dataset, f'train_fold_{fold_id}.csv')

            test_dataset = self._read_subset(dataset, f'test_fold_{fold_id}.csv')

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }
