from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset
from .base_model_selection import BaseModelSelection
from .utils import validate_partition_ids, write_partition_indices

__all__ = ['TrainValTestSplitCrossSubject']


class TrainValTestSplitCrossSubject(BaseModelSelection):
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
        validate_partition_ids(
            [self.train_subjects, self.val_subjects, self.test_subjects])
        super().__init__(split_path=split_path)

    def build(self, info: pd.DataFrame, split_path=None) -> None:
        write_partition_indices(
            info, 'subject_id',
            [self.train_subjects, self.val_subjects, self.test_subjects],
            self.split_path if split_path is None else split_path)

    def split(
            self, dataset: BaseDataset
    ) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        self._prepare(dataset, 'subject_id', files)
        yield tuple(self._read_subset(dataset, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'test_subjects': self.test_subjects,
            'split_path': self.split_path
        }
