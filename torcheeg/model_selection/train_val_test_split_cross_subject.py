import os
from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path
from ._three_way import (ensure_split_cache, read_index_subset,
                         validate_partition_ids, write_group_indices)


class TrainValTestSplitCrossSubject:
    r'''
    A tool class for cross-subject dataset splitting, to divide the dataset
    into training, validation, and test sets by predetermined subject ID lists.
    This is designed to align with evaluation protocols that use predefined
    subject partitions (e.g., CBraMod).

    .. code-block:: python

        from torcheeg.model_selection import TrainValTestSplitCrossSubject
        from torcheeg.datasets import FACEDDataset

        dataset = FACEDDataset(root_path='./Processed_data')
        splitter = TrainValTestSplitCrossSubject(
            train_subjects=list(range(0, 80)),
            val_subjects=list(range(80, 100)),
            test_subjects=list(range(100, 123)),
        )

        for train_dataset, val_dataset, test_dataset in splitter.split(dataset):
            ...

    The three ID lists must be nonempty, unique, disjoint and present in the
    dataset. IDs not assigned to any list are intentionally excluded. Original
    metadata dtypes are preserved, regardless of the DataFrame index labels.
    A cache is reused only for the same configuration and ordered sample IDs.
    Use a new :obj:`split_path` for older caches or changed partitions. Cache
    creation is atomic across processes; a failed creation can be retried.

    Args:
        train_subjects (list): List of subject IDs for the training set.
        val_subjects (list): List of subject IDs for the validation set.
        test_subjects (list): List of subject IDs for the test set.
        split_path (str): The path to data partition information. If the path
            exists, read the existing partition from the path. If the path does
            not exist, the current division method will be saved for next use.
            If set to None, a random path will be generated.
            (default: :obj:`None`)
    '''

    def __init__(self,
                 train_subjects: List,
                 val_subjects: List,
                 test_subjects: List,
                 split_path: Union[None, str] = None):
        self.train_subjects = list(train_subjects)
        self.val_subjects = list(val_subjects)
        self.test_subjects = list(test_subjects)
        validate_partition_ids([self.train_subjects, self.val_subjects, self.test_subjects])
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))

    def split_info_constructor(self, info: pd.DataFrame, split_path=None) -> None:
        write_group_indices(info, 'subject_id',
                            [self.train_subjects, self.val_subjects, self.test_subjects],
                            self.split_path if split_path is None else split_path)

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        configuration = {
            'splitter': self.__class__.__name__,
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'test_subjects': self.test_subjects,
        }
        ensure_split_cache(dataset.info, 'subject_id', self.split_path,
                           configuration, files,
                           lambda path: self.split_info_constructor(dataset.info, path))
        yield tuple(read_index_subset(dataset, self.split_path, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'test_subjects': self.test_subjects,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        values = ', '.join(f'{name}={value!r}' for name, value in self.repr_body.items())
        return f'{self.__class__.__name__}({values})'
