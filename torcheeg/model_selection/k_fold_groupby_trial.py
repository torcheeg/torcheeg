import os
import re
from typing import Iterator, Dict, List, Tuple, Union

import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import read_subset, write_indices

__all__ = ['KFoldGroupbyTrial']


class KFoldGroupbyTrial(BaseModelSelection):
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. A variant of :obj:`KFold`, where the data set is divided into k subsets at the dimension of trials, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    :obj:`KFoldGroupbyTrial` divides subsets at the dimension of trials. Take the first partition with :obj:`k=5` as an example, the first 80% of samples of each trial are used for training, and the last 20% of samples are used for testing. It is more consistent with real applications and can test the generalization of the model to a certain extent.

    .. image:: _static/splits/KFoldGroupbyTrial.svg
        :width: 100%
        :alt: The schematic diagram of KFoldGroupbyTrial
        :align: center

    |

    .. code-block:: python

        from torcheeg.model_selection import KFoldGroupbyTrial
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torch.utils.data import DataLoader

        cv = KFoldGroupbyTrial(n_splits=5, shuffle=False)
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
        train_infos = {}
        test_infos = {}
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            trial_ids = list(pd.unique(subject_info['trial_id']))
            for trial_id in trial_ids:
                trial_info = subject_info[subject_info['trial_id'] == trial_id]
                for i, (train_index,
                        test_index) in enumerate(self._k_fold.split(trial_info)):
                    train_info = trial_info.iloc[train_index]
                    test_info = trial_info.iloc[test_index]
                    if i not in train_infos:
                        train_infos[i] = []
                    if i not in test_infos:
                        test_infos[i] = []
                    train_infos[i].append(train_info)
                    test_infos[i].append(test_info)
        for i in train_infos.keys():
            train_info = pd.concat(train_infos[i], ignore_index=False)
            test_info = pd.concat(test_infos[i], ignore_index=False)
            write_indices(os.path.join(split_path, f'train_fold_{i}.csv'),
                          train_info.index.to_numpy())
            write_indices(os.path.join(split_path, f'test_fold_{i}.csv'),
                          test_info.index.to_numpy())

    @property
    def fold_ids(self) -> List:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self._prepare(dataset, ['subject_id', 'trial_id'])

        fold_ids = self.fold_ids

        for fold_id in fold_ids:

            train_dataset = read_subset(dataset, self.split_path,
                                        f'train_fold_{fold_id}.csv')

            test_dataset = read_subset(dataset, self.split_path,
                                       f'test_fold_{fold_id}.csv')

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
