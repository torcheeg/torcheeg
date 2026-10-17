import logging
import os
import re
from copy import copy
from typing import Iterator, List, Tuple, Union
from urllib.parse import quote, unquote

import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset

from ..utils import get_random_dir_path

log = logging.getLogger('torcheeg')


class LeaveOneSubjectOutIndex:
    r'''
    A tool class for leave-one-subject-out cross-validations, to divide the
    training set and the test set, commonly used to study model performance in
    the case of subject independent experiments. During each fold, experiments
    require testing on one subject and training on the other subjects. Unlike
    :obj:`LeaveOneSubjectOut`, the split is persisted as row indices and
    restored via :obj:`iloc`. The original metadata values and dtypes are
    preserved, including when the DataFrame has a nonconsecutive or duplicate
    index. Cached positions are valid only for the same samples in the same
    order; use a new :obj:`split_path` after regenerating or reordering a dataset.

    .. code-block:: python

        from torch.utils.data import DataLoader
        from torcheeg.model_selection import LeaveOneSubjectOutIndex

        cv = LeaveOneSubjectOutIndex()

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
        if 'subject_id' not in info or info['subject_id'].isna().any():
            raise ValueError('The dataset must contain nonmissing subject_id values.')
        # Persist row positions, rather than the labels of the DataFrame index.
        info = info.reset_index(drop=True)
        subjects = sorted(info['subject_id'].unique(), key=str)
        if len(subjects) < 2:
            raise ValueError('At least two subjects are required for leave-one-subject-out splitting.')
        if len(set(map(str, subjects))) != len(subjects):
            raise ValueError('Subject IDs must have distinct string representations.')

        for test_subject in subjects:
            test_mask = info['subject_id'] == test_subject
            train_indices = info[~test_mask].index.tolist()
            test_indices = info[test_mask].index.tolist()
            # Encode identifiers so a subject containing '/' stays in split_path.
            subject_key = quote(str(test_subject), safe='')

            pd.DataFrame({'index': train_indices}).to_csv(
                os.path.join(self.split_path, f'train_subject_{subject_key}.csv'),
                index=False)
            pd.DataFrame({'index': test_indices}).to_csv(
                os.path.join(self.split_path, f'test_subject_{subject_key}.csv'),
                index=False)

    @property
    def subjects(self) -> List:
        subjects = []
        for index_file in os.listdir(self.split_path):
            match = re.fullmatch(r'train_subject_(.+)\.csv', index_file)
            if match:
                subjects.append(unquote(match.group(1)))
        return sorted(subjects)

    def split(self, dataset: BaseDataset, subject=None) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        if (not os.path.exists(self.split_path)
                or not os.listdir(self.split_path)):
            log.info(
                f'📊 | Create the split of train and test set.'
            )
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path, exist_ok=True)
            self.split_info_constructor(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and test set, use existing split from {self.split_path}.'
            )
            log.info(
                f'💡 | If the dataset is re-generated, you need to re-generate the split of the dataset instead of using the previous split.'
            )

        subjects = self.subjects
        if not subjects or any(
                not os.path.isfile(os.path.join(
                    self.split_path, f'test_subject_{quote(local_subject, safe="")}.csv'))
                for local_subject in subjects):
            raise ValueError('Incomplete split cache; use a new split_path.')
        if subject is not None:
            subject = str(subject)
            if subject not in subjects:
                raise ValueError(f'The subject {subject} should be in the subject list {subjects}.')
            subjects = [subject]

        for test_subject in subjects:
            subject_key = quote(test_subject, safe='')
            train_indices = pd.read_csv(os.path.join(
                self.split_path, f'train_subject_{subject_key}.csv'))['index'].tolist()
            test_indices = pd.read_csv(os.path.join(
                self.split_path, f'test_subject_{subject_key}.csv'))['index'].tolist()

            train_dataset = copy(dataset)
            train_dataset.info = dataset.info.iloc[train_indices].reset_index(drop=True)

            test_dataset = copy(dataset)
            test_dataset.info = dataset.info.iloc[test_indices].reset_index(drop=True)

            yield train_dataset, test_dataset
