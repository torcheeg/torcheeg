import os
from typing import Dict, Iterator, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset
from .base_model_selection import BaseModelSelection
from .utils import create_subset, read_indices, write_partitions

__all__ = ['TrainValTestSplitGroupbyFolder']


class TrainValTestSplitGroupbyFolder(BaseModelSelection):
    r'''
    A tool class for datasets where the train/validation/test split is
    determined by the source folder (e.g., ``Training set/``,
    ``Validation set/``, ``Test set/``). The split is read from a ``split``
    column in the dataset's info DataFrame, whose values are ``'train'``,
    ``'val'``, and ``'test'``.

    .. code-block:: python

        from torcheeg.model_selection import TrainValTestSplitGroupbyFolder

        splitter = TrainValTestSplitGroupbyFolder()
        for train_dataset, val_dataset, test_dataset in splitter.split(dataset):
            ...

    Args:
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
        timeout (float): Lock wait in seconds; -1 waits indefinitely.
            (default: :obj:`300.0`)
    '''

    def __init__(self,
                 split_path: Union[None, str] = None,
                 timeout: float = 300.0) -> None:
        super().__init__(split_path=split_path, timeout=timeout)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        if not info['split'].isin(['train', 'val', 'test']).all():
            raise ValueError(
                "The split column must contain only 'train', 'val' and 'test'.")
        write_partitions(info, 'split', [['train'], ['val'], ['test']],
                         self.split_path if split_path is None else split_path)

    def split(
            self, dataset: BaseDataset
    ) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        self._prepare(dataset, 'split', files)
        subsets = []
        for name in files:
            indices = read_indices(os.path.join(self.split_path, name), len(dataset))
            subsets.append(create_subset(dataset, indices))
        yield tuple(subsets)

    @property
    def repr_body(self) -> Dict:
        return {'split_path': self.split_path, 'timeout': self.timeout}
