import os
import re
from typing import List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import write_indices

__all__ = ['LeaveOneSubjectOut']


class LeaveOneSubjectOut(BaseModelSelection):
    r'''
    A tool class for leave-one-subject-out cross-validations, to divide the training set and the test set, commonly used to study model performance in the case of subject independent experiments. During each fold, experiments require testing on one subject and training on the other subjects.

    .. image:: _static/splits/LeaveOneSubjectOut.svg
        :width: 100%
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

        super().__init__(split_path=split_path)

    def build(self, info: pd.DataFrame, split_path=None) -> None:
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        subjects = list(set(info['subject_id']))
        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)
            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])
            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]
            write_indices(os.path.join(split_path, f'train_subject_{test_subject}.csv'),
                          train_info.index.to_numpy())
            write_indices(os.path.join(split_path, f'test_subject_{test_subject}.csv'),
                          test_info.index.to_numpy())

    @property
    def subjects(self) -> List:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_subject(indice_file):
            return re.findall(r'subject_(.*).csv', indice_file)[0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    def split(self,
              dataset: BaseDataset,
              subject=None) -> Tuple[BaseDataset, BaseDataset]:
        self._prepare(dataset, ['subject_id'])

        subjects = self.subjects
        if subject is not None:
            train_dataset = self._read_subset(dataset, f'train_subject_{subject}.csv')

            test_dataset = self._read_subset(dataset, f'test_subject_{subject}.csv')

            yield train_dataset, test_dataset
        else:
            for subject in subjects:

                train_dataset = self._read_subset(dataset,
                                                  f'train_subject_{subject}.csv')

                test_dataset = self._read_subset(dataset, f'test_subject_{subject}.csv')

                yield train_dataset, test_dataset
