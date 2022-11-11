import os
import re
from copy import copy
from typing import List, Tuple

import pandas as pd
from torcheeg.datasets.module.base_dataset import BaseDataset


class LeaveOneSubjectOut:
    r'''
    A tool class for leave-one-subject-out cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject independent experiments. During each fold, experiments require testing on one subject and training on the other subjects.

    .. image:: _static/LeaveOneSubjectOut.png
        :alt: The schematic diagram of LeaveOneSubjectOut
        :align: center

    |
    
    .. code-block:: python

        cv = LeaveOneSubjectOut('./split')
        dataset = DEAPDataset(io_path=f'./deap',
                              root_path='./data_preprocessed_python',
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
        split_path (str): The path to data partition information. If the path exists, read the existing partition from the path. If the path does not exist, the current division method will be saved for next use. (default: :obj:`./split/leave_one_subject_out`)
    '''
    def __init__(self, split_path: str = './split/leave_one_subject_out'):
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
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)

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
