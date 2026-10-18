import hashlib
import json
import os
import shutil
import tempfile
from copy import copy
from typing import Dict, Iterator, List, Tuple, Union
from urllib.parse import quote

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from torcheeg.datasets.module.base_dataset import BaseDataset
from ..utils import get_random_dir_path


_CACHE_TIMEOUT = 300.0
_MANIFEST = '_split_manifest.json'


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_file(path):
    with open(path, 'rb') as file:
        return _hash_bytes(file.read())


def _fingerprint_dataset(info, column):
    # Index labels do not affect iloc positions. Include sample identities when
    # available so reordered datasets cannot silently reuse old positions.
    columns = [column] if isinstance(column, str) else list(column)
    columns += [name for name in ('_record_id', 'clip_id')
                if name in info and name not in columns]
    header = [(name, str(info[name].dtype)) for name in columns]
    typed_values = pd.DataFrame({
        name: [(type(value).__name__, value) for value in info[name].tolist()]
        for name in columns})
    hashes = pd.util.hash_pandas_object(typed_values, index=False).to_numpy()
    return _hash_bytes(json.dumps(header).encode() + hashes.tobytes())


def _validate_split_cache(split_path, expected, files):
    try:
        with open(os.path.join(split_path, _MANIFEST)) as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError('Corrupt split cache; use a new split_path.')
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('Split parameters or dataset order have changed; use a new split_path.')
        checksums = manifest['files']
        if set(checksums) != set(files):
            raise ValueError('Incomplete split cache; use a new split_path.')
        for name in files:
            if _hash_file(os.path.join(split_path, name)) != checksums[name]:
                raise ValueError('Corrupt split cache; use a new split_path.')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Incomplete or corrupt split cache; use a new split_path.') from error


def _ensure_split_cache(info, columns, split_path, configuration, files, writer):
    r'''Create or reuse a complete split cache under a process-safe lock.'''
    columns = [columns] if isinstance(columns, str) else list(columns)
    for name in columns:
        if name not in info or info[name].isna().any():
            raise ValueError(f'The dataset must contain nonmissing {name} values.')
    split_path = os.path.abspath(os.fspath(split_path))
    expected = {'version': 1, 'dataset': _fingerprint_dataset(info, columns),
                'configuration': _hash_bytes(json.dumps(configuration, sort_keys=True,
                    default=lambda value: value.item() if isinstance(value, np.generic)
                    else str(value)).encode())}
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    try:
        with FileLock(split_path + '.lock', timeout=_CACHE_TIMEOUT):
            if os.path.isfile(os.path.join(split_path, _MANIFEST)):
                _validate_split_cache(split_path, expected, files)
                return
            if os.path.exists(split_path) and (not os.path.isdir(split_path)
                                              or os.listdir(split_path)):
                raise ValueError('Incomplete split cache; use a new split_path.')
            staging_path = tempfile.mkdtemp(prefix='.torcheeg-split-',
                                            dir=os.path.dirname(split_path))
            try:
                writer(staging_path)
                manifest = dict(expected)
                manifest['files'] = {name: _hash_file(os.path.join(staging_path, name))
                                     for name in files}
                with open(os.path.join(staging_path, _MANIFEST), 'w') as file:
                    json.dump(manifest, file, sort_keys=True)
                if os.path.isdir(split_path):
                    os.rmdir(split_path)
                os.replace(staging_path, split_path)
            finally:
                shutil.rmtree(staging_path, ignore_errors=True)
    except Timeout as error:
        raise RuntimeError(f'Timeout waiting for split creation at {split_path}.') from error


def _read_split_subset(dataset, split_path, filename):
    r'''Restore a subset without converting metadata through CSV.'''
    frame = pd.read_csv(os.path.join(split_path, filename))
    if list(frame.columns) != ['index'] or frame['index'].dtype.kind not in 'iu':
        raise ValueError('Split cache must contain integer row positions.')
    indices = frame['index'].to_numpy()
    if (indices.size == 0 or (indices < 0).any()
            or (indices >= len(dataset)).any()
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Split cache contains invalid row positions.')
    subset = copy(dataset)
    subset.info = dataset.info.iloc[indices].reset_index(drop=True)
    return subset

class TrainTestSplitPerSubjectGroupbyTrial:
    r'''
    Split samples within each trial into training and test sets for each subject.

    Each yielded pair contains one subject. Each trial contributes
    :obj:`int(n_samples * train_size)` samples to training and the remaining
    samples to testing. Every trial must contribute at least one sample to
    each set. Without shuffling, the first samples of each trial form training.

    This is a subject-dependent, within-trial evaluation. Trials appear in both
    sets, so this does not measure generalization to unseen trials. Adjacent or
    overlapping windows may remain correlated across sets. Use
    :obj:`train_test_split_per_subject_cross_trial` for trial isolation.

    .. code-block:: python

        from torcheeg.model_selection import TrainTestSplitPerSubjectGroupbyTrial

        splitter = TrainTestSplitPerSubjectGroupbyTrial(
            train_size=0.8, shuffle=True, random_state=42)
        for train_dataset, test_dataset in splitter.split(dataset):
            ...

        # Limit iteration to one subject using its metadata ID.
        subject = dataset.info['subject_id'].iloc[0]
        train_dataset, test_dataset = next(splitter.split(dataset, subject=subject))

    Metadata types are preserved. Cached positions are reused only for the same
    parameters and ordered subject, trial and sample IDs. Use a new
    :obj:`split_path` when these change or when migrating an older cache.
    Cache creation is atomic across processes. Shuffling uses local random
    state and does not change NumPy's global random state.

    Args:
        train_size (float): Training proportion, strictly between zero and one.
            (default: :obj:`0.8`)
        shuffle (bool): Whether to shuffle samples within each trial.
            (default: :obj:`False`)
        random_state (int, optional): Seed for local shuffling.
            (default: :obj:`None`)
        split_path (str, optional): Directory for reusable split information.
            If None, generate a random path. (default: :obj:`None`)
    '''

    def __init__(self,
                 train_size: float = 0.8,
                 shuffle: bool = False,
                 random_state: Union[int, None] = None,
                 split_path: Union[None, str] = None):
        if (isinstance(train_size, (bool, np.bool_))
                or not isinstance(train_size, (int, float, np.number))
                or not 0 < train_size < 1):
            raise ValueError('train_size must be strictly between zero and one.')
        if not isinstance(shuffle, (bool, np.bool_)):
            raise ValueError('shuffle must be a boolean.')
        if (random_state is not None
                and (isinstance(random_state, (bool, np.bool_))
                     or not isinstance(random_state, (int, np.integer))
                     or not 0 <= random_state <= 2 ** 32 - 1)):
            raise ValueError('random_state must be None or an integer in [0, 2**32 - 1].')
        self.train_size = float(train_size)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))

    @staticmethod
    def _subject_ids(info: pd.DataFrame) -> List[str]:
        if info.empty:
            raise ValueError('The dataset must not be empty.')
        for column in ('subject_id', 'trial_id'):
            if column not in info or info[column].isna().any():
                raise ValueError(f'The dataset must contain nonmissing {column} values.')
        subjects = list(pd.unique(info['subject_id']))
        names = [str(subject) for subject in subjects]
        if len(set(names)) != len(names):
            raise ValueError('Subject IDs must have distinct string representations.')
        return names

    @staticmethod
    def _filename(subject: str, partition: str) -> str:
        return f'{partition}_subject_{quote(subject, safe="")}.csv'

    def split_info_constructor(self, info: pd.DataFrame, split_path=None) -> None:
        split_path = self.split_path if split_path is None else split_path
        subjects = self._subject_ids(info)
        rng = np.random.RandomState(self.random_state)
        os.makedirs(split_path, exist_ok=True)
        subject_values = pd.unique(info['subject_id'])
        for subject, subject_value in zip(subjects, subject_values):
            subject_positions = np.flatnonzero(
                info['subject_id'].eq(subject_value).to_numpy())
            subject_info = info.iloc[subject_positions]
            train_indices, test_indices = [], []
            for trial in pd.unique(subject_info['trial_id']):
                positions = subject_positions[
                    subject_info['trial_id'].eq(trial).to_numpy()]
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
                pd.DataFrame({'index': indices}).to_csv(
                    os.path.join(split_path, self._filename(subject, name)), index=False)
        with open(os.path.join(split_path, '_subjects.json'), 'w') as file:
            json.dump(subjects, file)

    @property
    def subjects(self) -> List[str]:
        r'''Subject ID strings in first-occurrence order, after cache creation.'''
        with open(os.path.join(self.split_path, '_subjects.json')) as file:
            return json.load(file)

    def split(self, dataset: BaseDataset,
              subject: Union[int, str, None] = None
              ) -> Iterator[Tuple[BaseDataset, BaseDataset]]:
        subjects = self._subject_ids(dataset.info)
        if subject is not None and str(subject) not in subjects:
            raise ValueError(f'The subject {subject!r} must be in {subjects}.')
        files = ['_subjects.json'] + [self._filename(value, partition)
                for value in subjects for partition in ('train', 'test')]
        configuration = {'splitter': self.__class__.__name__,
                         'train_size': self.train_size, 'shuffle': self.shuffle,
                         'random_state': self.random_state}
        _ensure_split_cache(dataset.info, ['subject_id', 'trial_id'], self.split_path,
                           configuration, files,
                           lambda path: self.split_info_constructor(dataset.info, path))
        for value in subjects:
            if subject is None or str(subject) == value:
                yield tuple(_read_split_subset(dataset, self.split_path,
                    self._filename(value, partition)) for partition in ('train', 'test'))

    @property
    def repr_body(self) -> Dict:
        return {'train_size': self.train_size, 'shuffle': self.shuffle,
                'random_state': self.random_state, 'split_path': self.split_path}

    def __repr__(self) -> str:
        values = ', '.join(f'{name}={value!r}' for name, value in self.repr_body.items())
        return f'{self.__class__.__name__}({values})'
