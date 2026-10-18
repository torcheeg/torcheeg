from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset
from .base_model_selection import BaseModelSelection
from .utils import validate_partition_ids, write_partition_indices

__all__ = ['TrainValTestSplitCrossTrial']


class TrainValTestSplitCrossTrial(BaseModelSelection):
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
    dataset. IDs not assigned to any list are intentionally excluded.

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
                 split_path: Union[None, str] = None) -> None:
        self.train_trials = list(train_trials)
        self.val_trials = list(val_trials)
        self.test_trials = list(test_trials)
        validate_partition_ids([self.train_trials, self.val_trials, self.test_trials])
        super().__init__(split_path=split_path)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        write_partition_indices(info, 'trial_id',
                                [self.train_trials, self.val_trials, self.test_trials],
                                self.split_path if split_path is None else split_path)

    def split(
            self, dataset: BaseDataset
    ) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        self._prepare(dataset, 'trial_id', files)
        yield tuple(self._read_subset(dataset, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {
            'train_trials': self.train_trials,
            'val_trials': self.val_trials,
            'test_trials': self.test_trials,
            'split_path': self.split_path
        }
