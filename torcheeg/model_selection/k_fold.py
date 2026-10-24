import os
import re
from typing import Iterator, Dict, Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import create_subset, read_indices, write_indices

class KFold(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. One of the most commonly used data partitioning methods, where the data set is divided into k subsets, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    :obj:`KFold` divides subsets without grouping. It means that during random sampling, adjacent signal samples may be assigned to the training set and the test set, respectively. When random sampling is not used, some subjects are not included in the training set. If you think these situations shouldn't happen, consider using :obj:`KFoldPerSubjectGroupbyTrial` or :obj:`KFoldGroupbyTrial`.

    .. image:: _static/splits/KFold.svg
        :width: 100%
        :alt: The schematic diagram of KFold
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFold
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        cv = KFold(n_splits=5, shuffle=True)
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

        for train_dataset, test_dataset in cv.split(dataset):
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...
    
    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. (default: :obj:`False`)
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, which controls the randomness of each fold. Otherwise, this parameter has no effect. (default: :obj:`None`)
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
    '''

    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: Union[None, str] = None) -> None:

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(split_path=split_path)

        self._k_fold = model_selection.KFold(n_splits=n_splits,
                                             shuffle=shuffle,
                                             random_state=random_state)

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        for fold_id, (train_index, test_index) in enumerate(self._k_fold.split(info)):
            write_indices(os.path.join(split_path, f'train_fold_{fold_id}.csv'),
                          train_index)
            write_indices(os.path.join(split_path, f'test_fold_{fold_id}.csv'),
                          test_index)

    @property
    def fold_ids(self):
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self._prepare(dataset, [])

        fold_ids = self.fold_ids

        for fold_id in fold_ids:

            train_indices = read_indices(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'),
                len(dataset))
            train_dataset = create_subset(dataset, train_indices)

            test_indices = read_indices(
                os.path.join(self.split_path, f'test_fold_{fold_id}.csv'), len(dataset))
            test_dataset = create_subset(dataset, test_indices)

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }
