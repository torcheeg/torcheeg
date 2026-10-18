import os
import re
from typing import List, Tuple, Union, Dict

import pandas as pd
from sklearn import model_selection
from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import write_indices

__all__ = ['KFoldPerSubject']


class KFoldPerSubject(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject dependent experiments. Experiments were performed separately for each subject, where the data of the subject is divided into k subsets, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/splits/KFoldPerSubject.svg
        :width: 100%
        :alt: The schematic diagram of KFoldPerSubject
        :align: center

    |

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.model_selection import KFoldPerSubject
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubject(n_splits=5, shuffle=True)
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

        for train_dataset, test_dataset in cv.split(dataset):
            # The total number of experiments is the number subjects multiplied by K
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...
    
    :obj:`KFoldPerSubject` allows the user to specify the index of the subject of interest, when the user need to report the performance on each subject.

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.model_selection import KFoldPerSubject
        from torcheeg.utils import DataLoader

        cv = KFoldPerSubject(n_splits=5, shuffle=True)
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

        for train_dataset, test_dataset in cv.split(dataset, subject=1):
            # k-fold cross-validation for subject 1
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: Union[None, str] = None):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(split_path=split_path)

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def build(self, info: pd.DataFrame, split_path=None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        subjects = list(set(info['subject_id']))
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            for i, (train_index,
                    test_index) in enumerate(self.k_fold.split(subject_info)):
                train_info = subject_info.iloc[train_index]
                test_info = subject_info.iloc[test_index]
                write_indices(
                    os.path.join(split_path, f'train_subject_{subject}_fold_{i}.csv'),
                    train_info.index.to_numpy())
                write_indices(
                    os.path.join(split_path, f'test_subject_{subject}_fold_{i}.csv'),
                    test_info.index.to_numpy())

    @property
    def subjects(self) -> List:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self,
              dataset: BaseDataset,
              subject: Union[int, None] = None) -> Tuple[BaseDataset, BaseDataset]:
        self._prepare(dataset, ['subject_id'])

        subjects = self.subjects
        fold_ids = self.fold_ids

        if not subject is None:
            assert str(
                subject
            ) in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (not subject is None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:

                train_dataset = self._read_subset(
                    dataset, f'train_subject_{local_subject}_fold_{fold_id}.csv')

                test_dataset = self._read_subset(
                    dataset, f'test_subject_{local_subject}_fold_{fold_id}.csv')

                yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }
