import logging
import os
import re
from copy import copy
from typing import List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path

log = logging.getLogger('torcheeg')


class LeaveOneSubjectOut:
    r'''
    A tool class for leave-one-subject-out cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject independent experiments. During each fold, experiments require testing on one subject and training on the other subjects.

    .. image:: _static/LeaveOneSubjectOut.png
        :alt: The schematic diagram of LeaveOneSubjectOut
        :align: center

    |
    
    .. code-block:: python

        from torcheeg.model_selection import LeaveOneSubjectOut
        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.utils import DataLoader

        cv = LeaveOneSubjectOut()
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
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
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. If set to None, a random path will be generated. (default: :obj:`None`)
    '''

    def __init__(self, split_path: Union[None, str] = None):
        if split_path is None:
            split_path = get_random_dir_path(dir_prefix='model_selection')

        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))

        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)

            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])

            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]

            train_info.to_csv(os.path.join(self.split_path,
                                           f'train_subject_{test_subject}.csv'),
                              index=False)
            test_info.to_csv(os.path.join(self.split_path,
                                          f'test_subject_{test_subject}.csv'),
                             index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*).csv', indice_file)[0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects

        for subject in subjects:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_subject_{subject}.csv'))
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_subject_{subject}.csv'))

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset
