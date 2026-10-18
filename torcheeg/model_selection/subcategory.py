import os
import json
from urllib.parse import quote, unquote

import numpy as np
from typing import Iterator, Dict, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import write_indices

__all__ = ['Subcategory']


class Subcategory(BaseModelSelection):
    r'''
    A tool class for separating out subsets of specified categories, often used to extract data for a certain type of paradigm, or for a certain type of task. Each subset in the formed subset list contains only one type of data.
    
    Common usage:

    .. code-block:: python

        from torcheeg.datasets import M3CVDataset
        from torcheeg.model_selection import Subcategory
        from torcheeg import transforms
        from torch.utils.data import DataLoader
        
        cv = Subcategory()
        dataset = M3CVDataset(root_path='./aistudio',
                              online_transform=transforms.Compose(
                                  [transforms.To2d(),
                                   transforms.ToTensor()]),
                              label_transform=transforms.Compose([
                                  transforms.Select('subject_id'),
                                  transforms.StringToInt()
                              ]))
        for subdataset in cv.split(dataset):
            loader = DataLoader(subdataset)
            ...
    
    TorchEEG supports the division of training and test sets within each subset after dividing the data into subsets. The sample code is as follows:

    .. code-block:: python

        cv = Subcategory()
        dataset = M3CVDataset(root_path='./aistudio',
                              online_transform=transforms.Compose(
                                  [transforms.To2d(),
                                   transforms.ToTensor()]),
                              label_transform=transforms.Compose([
                                  transforms.Select('subject_id'),
                                  transforms.StringToInt()
                              ]))
        for i, subdataset in enumerate(cv.split(dataset)):
            train_dataset, test_dataset = train_test_split(dataset=subdataset, split_path=f'./split{i}')

            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)
            ...
    
    For the already divided training and testing sets, TorchEEG recommends using two :obj:`Subcategory` to extract their subcategories respectively. On this basis, the :obj:`zip` function can be used to combine the subsets. It is worth noting that it is necessary to ensure that the training and test sets have the same number and variety of classes.

    .. code-block:: python

        train_cv = Subcategory()
        train_dataset = M3CVDataset(root_path='./aistudio',
                                    online_transform=transforms.Compose(
                                        [transforms.To2d(),
                                         transforms.ToTensor()]),
                                    label_transform=transforms.Compose([
                                        transforms.Select('subject_id'),
                                        transforms.StringToInt()
                                    ]))

        val_cv = Subcategory()
        val_dataset = M3CVDataset(root_path='./aistudio',
                                  subset='Calibration',
                                  num_channel=65,
                                  online_transform=transforms.Compose(
                                      [transforms.To2d(),
                                       transforms.ToTensor()]),
                                  label_transform=transforms.Compose([
                                      transforms.Select('subject_id'),
                                      transforms.StringToInt()
                                  ]))

        for train_dataset, val_dataset in zip(train_cv.split(train_dataset), val_cv.split(val_dataset)):
            train_loader = DataLoader(train_dataset)
            val_loader = DataLoader(val_dataset)
            ...

    Args:
        criteria (str): The classification criteria according to which we extract subsets of data for the including categories. (default: :obj:`'task'`)
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
        timeout (float): Lock wait in seconds; -1 waits indefinitely.
            (default: :obj:`300.0`)
    '''

    def __init__(self,
                 criteria: str = 'task',
                 split_path: Union[None, str] = None,
                 timeout: float = 300.0) -> None:

        self.criteria = criteria
        super().__init__(split_path=split_path, timeout=timeout)

    @staticmethod
    def _filename(category) -> str:
        if isinstance(category, np.generic):
            category = category.item()
        return quote(json.dumps(category), safe='') + '.csv'

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        if self.criteria not in info:
            raise ValueError(f'Unsupported criteria {self.criteria}.')
        category_list = list(pd.unique(info[self.criteria]))
        for category in category_list:
            subset_info = info[info[self.criteria] == category]
            write_indices(os.path.join(split_path, self._filename(category)),
                          subset_info.index.to_numpy())

    @property
    def category_list(self):
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        category_list = [json.loads(unquote(name[:-4])) for name in indice_files]
        try:
            category_list.sort()
        except TypeError:
            category_list.sort(key=lambda value: (type(value).__name__, str(value)))

        return category_list

    def split(self, dataset: BaseDataset) -> Iterator[BaseDataset]:
        self._prepare(dataset, [self.criteria])

        category_list = self.category_list

        for category in category_list:

            subset_dataset = self._read_subset(dataset, self._filename(category))

            yield subset_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'criteria': self.criteria,
            'split_path': self.split_path,
            'timeout': self.timeout
        }
