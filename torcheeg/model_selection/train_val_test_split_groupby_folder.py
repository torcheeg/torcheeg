from typing import Dict, Iterator, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset
from .base_model_selection import BaseModelSelection
from .utils import write_partition_indices

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

    Each row must have one of the three split values, and all three subsets
    must be nonempty. Folder labels determine the split; this does not itself
    guarantee that subjects or trials are disjoint across the source folders.
    Metadata dtypes are preserved. Existing caches must match the configuration
    and ordered sample IDs; use a new :obj:`split_path` after changing the data.

    Args:
        split_path (str): The path to data partition information. If the path
            exists, read the existing partition from the path. If the path does
            not exist, the current division method will be saved for next use.
            If set to None, a random path will be generated.
            (default: :obj:`None`)
    '''

    def __init__(self, split_path: Union[None, str] = None) -> None:
        super().__init__(split_path=split_path)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        if not info['split'].isin(['train', 'val', 'test']).all():
            raise ValueError(
                "The split column must contain only 'train', 'val' and 'test'.")
        write_partition_indices(info, 'split', [['train'], ['val'], ['test']],
                                self.split_path if split_path is None else split_path)

    def split(
            self, dataset: BaseDataset
    ) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        self._prepare(dataset, 'split', files)
        yield tuple(self._read_subset(dataset, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {'split_path': self.split_path}
