import os
import re
from urllib.parse import quote, unquote
from typing import Iterator, Dict, List, Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import create_subset, read_indices, validate_group_names, write_indices

__all__ = ['KFoldPerSubjectGroupbyTrial']


class KFoldPerSubjectGroupbyTrial(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject dependent experiments. Experiments were performed separately for each subject, where the data for all trials of the subject is divided into k subsets at the trial dimension, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    .. image:: _static/splits/KFoldPerSubjectGroupbyTrial.svg
        :width: 100%
        :alt: The schematic diagram of KFoldPerSubjectGroupbyTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        cv = KFoldPerSubjectGroupbyTrial(n_splits=5, shuffle=True)
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
    
    :obj:`KFoldPerSubjectGroupbyTrial` allows the user to specify the index of the subject of interest, when the user need to report the performance on each subject.

    .. code-block:: python

        from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        cv = KFoldPerSubjectGroupbyTrial(n_splits=5, shuffle=True)
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
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
        timeout (float): Lock wait in seconds; -1 waits indefinitely.
            (default: :obj:`300.0`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[int, None] = None,
                 split_path: Union[None, str] = None,
                 timeout: float = 300.0) -> None:

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(split_path=split_path, timeout=timeout)

        self._k_fold = model_selection.KFold(n_splits=n_splits,
                                             shuffle=shuffle,
                                             random_state=random_state)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        subjects = list(pd.unique(info['subject_id']))
        validate_group_names(subjects)
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            subject_train_infos = {}
            subject_test_infos = {}
            trial_ids = list(pd.unique(subject_info['trial_id']))
            for trial_id in trial_ids:
                trial_info = subject_info[subject_info['trial_id'] == trial_id]
                for i, (train_index,
                        test_index) in enumerate(self._k_fold.split(trial_info)):
                    train_info = trial_info.iloc[train_index]
                    test_info = trial_info.iloc[test_index]
                    if i not in subject_train_infos:
                        subject_train_infos[i] = []
                    if i not in subject_test_infos:
                        subject_test_infos[i] = []
                    subject_train_infos[i].append(train_info)
                    subject_test_infos[i].append(test_info)
            for i in subject_train_infos.keys():
                subject_train_info = pd.concat(subject_train_infos[i],
                                               ignore_index=False)
                subject_test_info = pd.concat(subject_test_infos[i], ignore_index=False)
                write_indices(
                    os.path.join(
                        split_path,
                        f'train_subject_{quote(str(subject), safe="")}_fold_{i}.csv'),
                    subject_train_info.index.to_numpy())
                write_indices(
                    os.path.join(
                        split_path,
                        f'test_subject_{quote(str(subject), safe="")}_fold_{i}.csv'),
                    subject_test_info.index.to_numpy())

    @property
    def subjects(self) -> List[str]:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_subject(indice_file):
            return unquote(
                re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][0])

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    @property
    def fold_ids(self) -> List[int]:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'subject_(.*)_fold_(\d*).csv', indice_file)[0][1])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(
        self,
        dataset: BaseDataset,
        subject: Union[int, str,
                       None] = None) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self._prepare(dataset, ['subject_id', 'trial_id'])

        subjects = self.subjects
        fold_ids = self.fold_ids

        if subject is not None:
            if str(subject) not in subjects:
                raise ValueError(
                    f'The subject should be in the subject list {subjects}.')

        for local_subject in subjects:
            if subject is not None and local_subject != str(subject):
                continue

            for fold_id in fold_ids:

                train_indices = read_indices(
                    os.path.join(
                        self.split_path,
                        f"train_subject_{quote(str(local_subject), safe='')}_fold_{fold_id}.csv"
                    ), len(dataset))
                train_dataset = create_subset(dataset, train_indices)

                test_indices = read_indices(
                    os.path.join(
                        self.split_path,
                        f"test_subject_{quote(str(local_subject), safe='')}_fold_{fold_id}.csv"
                    ), len(dataset))
                test_dataset = create_subset(dataset, test_indices)

                yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path,
            'timeout': self.timeout
        }
