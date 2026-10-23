import os
from typing import Dict, Iterator, List, Tuple, Union
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd

from torcheeg.datasets.module.base_dataset import BaseDataset
from .base_model_selection import BaseModelSelection
from .utils import create_subset, read_indices, validate_groups, write_indices

__all__ = ['TrainTestSplitPerSubjectGroupbyTrial']


class TrainTestSplitPerSubjectGroupbyTrial(BaseModelSelection):
    r'''
    Split samples within each trial into training and test sets for each subject.

    .. code-block:: python

        from torcheeg.model_selection import TrainTestSplitPerSubjectGroupbyTrial

        splitter = TrainTestSplitPerSubjectGroupbyTrial(
            train_size=0.8, shuffle=True, random_state=42)
        for train_dataset, test_dataset in splitter.split(dataset):
            ...

        # Limit iteration to one subject using its metadata ID.
        subject = dataset.info['subject_id'].iloc[0]
        train_dataset, test_dataset = next(splitter.split(dataset, subject=subject))

    Args:
        train_size (float): Training proportion, strictly between zero and one.
            (default: :obj:`0.8`)
        shuffle (bool): Whether to shuffle samples within each trial.
            (default: :obj:`False`)
        random_state (int, optional): Seed for local shuffling.
            (default: :obj:`None`)
        split_path (str, optional): Directory for saved splits.
            If None, generate a random path. (default: :obj:`None`)
    '''

    def __init__(self,
                 train_size: float = 0.8,
                 shuffle: bool = False,
                 random_state: Union[int, None] = None,
                 split_path: Union[None, str] = None) -> None:
        if isinstance(train_size,
                      (bool, np.bool_)) or not isinstance(train_size,
                                                          (int, float, np.number)):
            raise TypeError('train_size must be a real number.')
        if not 0 < train_size < 1:
            raise ValueError('train_size must be strictly between zero and one.')
        if not isinstance(shuffle, (bool, np.bool_)):
            raise TypeError('shuffle must be a boolean.')
        if random_state is not None:
            if isinstance(
                    random_state,
                (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
                raise TypeError('random_state must be None or an integer.')
            if not 0 <= random_state <= 2**32 - 1:
                raise ValueError('random_state must be in [0, 2**32 - 1].')
        self.train_size = float(train_size)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        super().__init__(split_path=split_path)

    @staticmethod
    def _filename(subject: str, partition: str) -> str:
        return f'{partition}_subject_{quote(subject, safe="")}.csv'

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        split_path = self.split_path if split_path is None else split_path
        subject_values = pd.unique(info['subject_id'])
        subjects = validate_groups(subject_values)
        rng = np.random.RandomState(self.random_state)
        os.makedirs(split_path, exist_ok=True)
        for subject, subject_value in zip(subjects, subject_values):
            subject_positions = np.flatnonzero(
                info['subject_id'].eq(subject_value).to_numpy())
            subject_info = info.iloc[subject_positions]
            train_indices, test_indices = [], []
            for trial in pd.unique(subject_info['trial_id']):
                positions = subject_positions[subject_info['trial_id'].eq(
                    trial).to_numpy()]
                train_size = int(len(positions) * self.train_size)
                if not 0 < train_size < len(positions):
                    raise ValueError(
                        f'Subject {subject!r}, trial {trial!r} must contribute '
                        'at least one training and one test sample.')
                if self.shuffle:
                    positions = rng.permutation(positions)
                train_indices.extend(positions[:train_size])
                test_indices.extend(positions[train_size:])
            for name, indices in (('train', train_indices), ('test', test_indices)):
                write_indices(os.path.join(split_path, self._filename(subject, name)),
                              indices)

    @property
    def subjects(self) -> List[str]:
        return sorted(
            unquote(name[len('train_subject_'):-4])
            for name in os.listdir(self.split_path)
            if name.startswith('train_subject_') and name.endswith('.csv'))

    def split(
        self,
        dataset: BaseDataset,
        subject: Union[int, str,
                       None] = None) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        self.check(dataset, ['subject_id', 'trial_id'])
        subjects = validate_groups(pd.unique(dataset.info['subject_id']))
        if subject is not None and str(subject) not in subjects:
            raise ValueError(f'The subject {subject!r} must be in {subjects}.')
        files = [
            self._filename(value, partition) for value in subjects
            for partition in ('train', 'test')
        ]
        self._prepare(dataset, ['subject_id', 'trial_id'], files)
        for value in subjects:
            if subject is None or str(subject) == value:
                subsets = []
                for partition in ('train', 'test'):
                    indices = read_indices(
                        os.path.join(self.split_path, self._filename(value, partition)),
                        len(dataset))
                    subsets.append(create_subset(dataset, indices))
                yield tuple(subsets)

    @property
    def repr_body(self) -> Dict:
        return {
            'train_size': self.train_size,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }
