import os
import re
from copy import copy
from typing import Tuple, Union

import pandas as pd
from sklearn import model_selection
from torcheeg.datasets.module.base_dataset import BaseDataset


class KFoldDataset:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. One of the most commonly used data partitioning methods, where the data set is divided into k subsets, with one subset being retained as the test set and the remaining k-1 being used as training data. In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    :obj:`KFoldDataset` devides subsets at the dataset dimension. It means that during random sampling, adjacent signal samples may be assigned to the training set and the test set, respectively. When random sampling is not used, some subjects are not included in the training set. If you think these situations shouldn't happen, consider using :obj:`KFoldTrialPerSubject` or :obj`KFoldTrial`.

    .. code-block:: python

        cv = KFoldDataset(n_splits=5, shuffle=True, split_path='./split')
        dataset = DEAPDataset(io_path=f'./deap',
                              root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: x.unsqueeze(0))
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
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. (default: :obj:`./split/k_fold_dataset`)
    '''
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: str = './split/k_fold_dataset'):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        for fold_id, (train_index, test_index) in enumerate(self.k_fold.split(info)):
            train_info = info.iloc[train_index]
            test_info = info.iloc[test_index]

            train_info.to_csv(os.path.join(self.split_path, 'train_fold_{}.csv'.format(fold_id)), index=False)
            test_info.to_csv(os.path.join(self.split_path, 'test_fold_{}.csv'.format(fold_id)), index=False)

    @property
    def fold_ids(self):
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        return list(set(map(indice_file_to_fold_id, indice_files)))

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(os.path.join(self.split_path, 'train_fold_{}.csv'.format(fold_id)))
            test_info = pd.read_csv(os.path.join(self.split_path, 'test_fold_{}.csv'.format(fold_id)))

            trian_dataset = copy(dataset)
            trian_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield trian_dataset, test_dataset
