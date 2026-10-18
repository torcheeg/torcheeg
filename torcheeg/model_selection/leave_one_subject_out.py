import os
import re
from urllib.parse import quote, unquote
from typing import Iterator, List, Tuple, Union

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from .base_model_selection import BaseModelSelection
from .utils import create_subset, read_indices, validate_group_names, write_indices

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
        from torch.utils.data import DataLoader

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
        info = info.reset_index(drop=True)
        split_path = self.split_path if split_path is None else split_path
        os.makedirs(split_path, exist_ok=True)
        subjects = list(pd.unique(info['subject_id']))
        validate_group_names(subjects)
        if len(subjects) < 2:
            raise ValueError('LeaveOneSubjectOut requires at least two subjects.')
        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)
            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])
            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]
            write_indices(
                os.path.join(split_path,
                             f'train_subject_{quote(str(test_subject), safe="")}.csv'),
                train_info.index.to_numpy())
            write_indices(
                os.path.join(split_path,
                             f'test_subject_{quote(str(test_subject), safe="")}.csv'),
                test_info.index.to_numpy())

    @property
    def subjects(self) -> List[str]:
        indice_files = [
            name for name in os.listdir(self.split_path) if name.endswith('.csv')
        ]

        def indice_file_to_subject(indice_file):
            return unquote(re.findall(r'subject_(.*).csv', indice_file)[0])

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    def split(
        self,
        dataset: BaseDataset,
        subject: Union[int, str,
                       None] = None) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self._prepare(dataset, ['subject_id'])

        subjects = self.subjects
        if subject is not None:
            if str(subject) not in subjects:
                raise ValueError(
                    f'The subject should be in the subject list {subjects}.')
            train_indices = read_indices(
                os.path.join(self.split_path,
                             f"train_subject_{quote(str(subject), safe='')}.csv"),
                len(dataset))
            train_dataset = create_subset(dataset, train_indices)

            test_indices = read_indices(
                os.path.join(self.split_path,
                             f"test_subject_{quote(str(subject), safe='')}.csv"),
                len(dataset))
            test_dataset = create_subset(dataset, test_indices)

            yield train_dataset, test_dataset
        else:
            for subject in subjects:

                train_indices = read_indices(
                    os.path.join(self.split_path,
                                 f"train_subject_{quote(str(subject), safe='')}.csv"),
                    len(dataset))
                train_dataset = create_subset(dataset, train_indices)

                test_indices = read_indices(
                    os.path.join(self.split_path,
                                 f"test_subject_{quote(str(subject), safe='')}.csv"),
                    len(dataset))
                test_dataset = create_subset(dataset, test_indices)

                yield train_dataset, test_dataset
