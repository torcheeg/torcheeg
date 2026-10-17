import logging
import os
import re
from copy import copy
from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path

log = logging.getLogger('torcheeg')


class KFoldCrossSubjectIndex:
    r'''
    A tool class for k-fold cross-subject cross-validation. All subjects are
    partitioned into :obj:`n_splits` folds. In each iteration, one fold of
    subjects serves as the test set and the remaining folds serve as the
    training set. Unlike :obj:`KFoldCrossSubject`, the split is persisted as
    row indices and restored via :obj:`iloc`. The original metadata values and
    dtypes are
    preserved, including when the DataFrame has a nonconsecutive or duplicate
    index. Cached positions are valid only for the same samples in the same
    order; use a new :obj:`split_path` after regenerating or reordering a dataset.

    .. code-block:: python

        from torch.utils.data import DataLoader
        from torcheeg.model_selection import KFoldCrossSubjectIndex

        cv = KFoldCrossSubjectIndex(n_splits=10, shuffle=True, random_state=42)

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the subjects before splitting into batches. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the subjects, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        if 'subject_id' not in info or info['subject_id'].isna().any():
            raise ValueError('The dataset must contain nonmissing subject_id values.')
        # Persist row positions, rather than the labels of the DataFrame index.
        info = info.reset_index(drop=True)
        subjects = list(set(info['subject_id']))
        subjects.sort()

        for fold_id, (train_subject_indices, test_subject_indices) in enumerate(self.k_fold.split(subjects)):
            train_subjects = [subjects[i] for i in train_subject_indices]
            test_subjects = [subjects[i] for i in test_subject_indices]

            train_indices = info[info['subject_id'].isin(
                train_subjects)].index.tolist()
            test_indices = info[info['subject_id'].isin(
                test_subjects)].index.tolist()

            train_indices_df = pd.DataFrame({'index': train_indices})
            test_indices_df = pd.DataFrame({'index': test_indices})

            train_indices_df.to_csv(os.path.join(self.split_path,
                                                 f'train_fold_{fold_id}.csv'),
                                    index=False)
            test_indices_df.to_csv(os.path.join(self.split_path,
                                                f'test_fold_{fold_id}.csv'),
                                   index=False)

    @property
    def fold_ids(self) -> List[int]:
        fold_ids = set()
        for index_file in os.listdir(self.split_path):
            match = re.fullmatch(r'(?:train|test)_fold_(\d+)\.csv', index_file)
            if match:
                fold_ids.add(int(match.group(1)))
        return sorted(fold_ids)

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        if (not os.path.exists(self.split_path)
                or not os.listdir(self.split_path)):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path, exist_ok=True)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        fold_ids = self.fold_ids
        if not fold_ids or any(
                not os.path.isfile(os.path.join(
                    self.split_path, f'{split}_fold_{fold_id}.csv'))
                for fold_id in fold_ids for split in ('train', 'test')):
            raise ValueError('Incomplete split cache; use a new split_path.')

        for fold_id in fold_ids:
            train_indices_df = pd.read_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_indices_df = pd.read_csv(
                os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            train_indices = train_indices_df['index'].tolist()
            test_indices = test_indices_df['index'].tolist()

            train_dataset = copy(dataset)
            train_dataset.info = dataset.info.iloc[train_indices].reset_index(
                drop=True)

            test_dataset = copy(dataset)
            test_dataset.info = dataset.info.iloc[test_indices].reset_index(
                drop=True)

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            if i:
                format_string += ', '
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string
