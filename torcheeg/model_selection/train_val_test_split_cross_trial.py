import os
from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path
from ._three_way import (ensure_split_cache, read_index_subset,
                         validate_partition_ids, write_group_indices)


class TrainValTestSplitCrossTrial:
    r'''
    A tool class for cross-trial dataset splitting, to divide the dataset
    into training, validation, and test sets by predetermined trial ID lists.
    All subjects contribute to every split; the partition boundary lies across
    trials.  This is designed to align with evaluation protocols that assign
    fixed trial indices to each split (e.g., CBraMod SEED-V: trials 0-4 for
    training, 5-9 for validation, 10-14 for testing).

    .. code-block:: python

        from torcheeg.model_selection import TrainValTestSplitCrossTrial
        from torcheeg.datasets import SEEDVDataset

        dataset = SEEDVDataset(root_path='./EEG_raw')
        splitter = TrainValTestSplitCrossTrial(
            train_trials=list(range(0, 5)),
            val_trials=list(range(5, 10)),
            test_trials=list(range(10, 15)),
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
        train_trials (list): List of trial IDs for the training set.
        val_trials (list): List of trial IDs for the validation set.
        test_trials (list): List of trial IDs for the test set.
        split_path (str): The path to data partition information. If the path
            exists, read the existing partition from the path. If the path does
            not exist, the current division method will be saved for next use.
            If set to None, a random path will be generated.
            (default: :obj:`None`)
    '''

    def __init__(self,
                 train_trials: List,
                 val_trials: List,
                 test_trials: List,
                 split_path: Union[None, str] = None):
        self.train_trials = list(train_trials)
        self.val_trials = list(val_trials)
        self.test_trials = list(test_trials)
        validate_partition_ids([self.train_trials, self.val_trials, self.test_trials])
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))

    def split_info_constructor(self, info: pd.DataFrame, split_path=None) -> None:
        write_group_indices(info, 'trial_id',
                            [self.train_trials, self.val_trials, self.test_trials],
                            self.split_path if split_path is None else split_path)

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        configuration = {
            'splitter': self.__class__.__name__,
            'train_trials': self.train_trials,
            'val_trials': self.val_trials,
            'test_trials': self.test_trials,
        }
        ensure_split_cache(dataset.info, 'trial_id', self.split_path,
                           configuration, files,
                           lambda path: self.split_info_constructor(dataset.info, path))
        yield tuple(read_index_subset(dataset, self.split_path, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {
            'train_trials': self.train_trials,
            'val_trials': self.val_trials,
            'test_trials': self.test_trials,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        values = ', '.join(f'{name}={value!r}' for name, value in self.repr_body.items())
        return f'{self.__class__.__name__}({values})'
