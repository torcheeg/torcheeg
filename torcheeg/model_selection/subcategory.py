import os
import re
from copy import copy
from typing import Tuple, Dict

import pandas as pd
from torcheeg.datasets.module.base_dataset import BaseDataset


class Subcategory:
    r'''
    A tool class for separating out subsets of specified categories, often used to extract data for a certain type of paradigm, or for a certain type of task. Each subset in the formed subset list contains only one type of data.
    
    Common usage:

    .. code-block:: python

        cv = Subcategory(split_path='./split')
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
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

        cv = Subcategory(split_path='./split')
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
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

        train_cv = Subcategory(split_path='./split_train')
        train_dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose(
                                  [transforms.To2d(),
                                   transforms.ToTensor()]),
                              label_transform=transforms.Compose([
                                  transforms.Select('subject_id'),
                                  transforms.StringToInt()
                              ]))

        val_cv = Subcategory(split_path='./split_val')
        val_dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              subset='Calibration', num_channel=65,
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
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. (default: :obj:`'./split/k_fold_dataset'`)
    '''
    def __init__(self,
                 criteria: str = 'task',
                 split_path: str = './split/subcategory'):
        self.criteria = criteria
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        assert self.criteria in list(
            info.columns
        ), f'Unsupported criteria {self.criteria}, please select one of the following options {list(info.columns)}.'

        category_list = list(set(info[self.criteria]))

        for category in category_list:
            subset_info = info[info[self.criteria] == category]
            subset_info.to_csv(os.path.join(self.split_path, f'{category}.csv'),
                               index=False)

    @property
    def category_list(self):
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'(.*).csv', indice_file)[0])

        category_list = list(set(map(indice_file_to_fold_id, indice_files)))
        category_list.sort()

        return category_list

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)

        category_list = self.category_list

        for category in category_list:
            subset_info = pd.read_csv(
                os.path.join(self.split_path, f'{category}.csv'))

            subset_dataset = copy(dataset)
            subset_dataset.info = subset_info

            yield subset_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'criteria': self.criteria,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ', '
            # str param
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string