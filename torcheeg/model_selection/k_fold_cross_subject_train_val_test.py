import os
from numbers import Integral
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import (create_subset, read_indices, validate_partitions,
                    write_partitions)


class KFoldCrossSubjectTrainValTest(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validation at the subject level, additionally
    holding out a validation set in each fold. Subjects are partitioned into
    :obj:`n_folds` groups; for fold :obj:`i`, group :obj:`i` is used as the test
    set, group :obj:`(i + 1) % n_folds` as the validation set, and the remaining
    groups as the training set. This ensures the same subject never appears in
    more than one of the train, validation and test sets.

    .. code-block:: python

        from torcheeg.model_selection import KFoldCrossSubjectTrainValTest
        from torcheeg.datasets import DEAPDataset

        dataset = DEAPDataset(root_path='./data_preprocessed_python')
        cv = KFoldCrossSubjectTrainValTest(
            all_subjects=dataset.info['subject_id'].unique().tolist(), n_folds=5)
        for train_dataset, val_dataset, test_dataset in cv.split(dataset):
            ...

    Args:
        all_subjects (list): The list of all subject ids to be partitioned into folds.
        n_folds (int): The number of folds. (default: :obj:`5`)
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
        shuffle (bool): Whether to shuffle the subjects before splitting them into folds. (default: :obj:`True`)
        random_state (int): The random seed used to shuffle the subjects when :obj:`shuffle` is :obj:`True`. (default: :obj:`42`)
    '''

    def __init__(self,
                 all_subjects: List,
                 n_folds: int = 5,
                 split_path: Union[None, str] = None,
                 shuffle: bool = True,
                 random_state: int = 42) -> None:
        self.all_subjects = list(all_subjects)
        validate_partitions([self.all_subjects])
        if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, Integral):
            raise TypeError('n_folds must be an integer.')
        if not 3 <= n_folds <= len(self.all_subjects):
            raise ValueError(
                'n_folds must be between 3 and the number of subjects.')
        if not isinstance(shuffle, (bool, np.bool_)):
            raise TypeError('shuffle must be a boolean.')
        if random_state is not None:
            if isinstance(random_state,
                          (bool, np.bool_)) or not isinstance(random_state, Integral):
                raise TypeError('random_state must be None or an integer.')
            if not 0 <= random_state <= 2**32 - 1:
                raise ValueError('random_state must be in [0, 2**32 - 1].')
        self.n_folds = int(n_folds)
        super().__init__(split_path=split_path)
        self.shuffle = shuffle
        self.random_state = random_state

    def _make_folds(self):
        subjects = list(self.all_subjects)
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(subjects)
        return [subjects[index::self.n_folds] for index in range(self.n_folds)]

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        if set(info['subject_id']) != set(self.all_subjects):
            raise ValueError(
                'all_subjects must match the dataset subject_id values.')
        split_path = self.split_path if split_path is None else split_path
        folds = self._make_folds()
        for fold_id in range(self.n_folds):
            test_subjects = folds[fold_id]
            val_subjects = folds[(fold_id + 1) % self.n_folds]
            train_subjects = [
                subject for index, fold in enumerate(folds)
                if index not in (fold_id, (fold_id + 1) % self.n_folds)
                for subject in fold
            ]
            write_partitions(info, 'subject_id',
                             [train_subjects, val_subjects, test_subjects],
                             os.path.join(split_path, f'fold_{fold_id}'))

    def split(
            self, dataset: BaseDataset
    ) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = [
            f'fold_{fold_id}/{name}.csv' for fold_id in range(self.n_folds)
            for name in ('train', 'val', 'test')
        ]
        self._prepare(dataset, 'subject_id', files)
        for fold_id in range(self.n_folds):
            subsets = []
            for name in ('train', 'val', 'test'):
                indices = read_indices(
                    os.path.join(self.split_path,
                                 f'fold_{fold_id}/{name}.csv'),
                    len(dataset))
                subsets.append(create_subset(dataset, indices))
            yield tuple(subsets)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body,
                    all_subjects=self.all_subjects,
                    n_folds=self.n_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state)
