import hashlib
import json
import os
import shutil
import tempfile
from copy import copy
from numbers import Integral
from typing import Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from torcheeg.datasets.module.base_dataset import BaseDataset
from ..utils import get_random_dir_path


_CACHE_TIMEOUT = 300.0
_MANIFEST = '_split_manifest.json'


def _validate_partition_ids(partitions):
    r'''Reject empty, repeated or overlapping train/validation/test IDs.'''
    seen = set()
    for ids in partitions:
        if not ids or pd.Series(ids, dtype=object).isna().any():
            raise ValueError('Partition IDs must be nonempty and nonmissing.')
        try:
            values = set(ids)
        except TypeError as error:
            raise ValueError('Partition IDs must be hashable.') from error
        if len(values) != len(ids):
            raise ValueError('Partition IDs must not contain duplicates.')
        if seen & values:
            raise ValueError('Train, validation and test IDs must be disjoint.')
        seen.update(values)


def _write_partition_indices(info, column, partitions, split_path):
    r'''Write positions in the original row order without persisting metadata.'''
    _validate_partition_ids(partitions)
    requested = set().union(*(set(ids) for ids in partitions))
    if requested - set(info[column]):
        raise ValueError(f'Unknown {column} values in the partition IDs.')
    os.makedirs(split_path, exist_ok=True)
    for name, ids in zip(('train', 'val', 'test'), partitions):
        positions = np.flatnonzero(info[column].isin(ids).to_numpy())
        pd.DataFrame({'index': positions}).to_csv(
            os.path.join(split_path, f'{name}.csv'), index=False)


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

class KFoldCrossSubjectTrainValTest:
    r'''
    A tool class for k-fold cross-validation at the subject level, additionally
    holding out a validation set in each fold. Subjects are partitioned into
    :obj:`n_folds` groups; for fold :obj:`i`, group :obj:`i` is used as the test
    set, group :obj:`(i + 1) % n_folds` as the validation set, and the remaining
    groups as the training set. This ensures the same subject never appears in
    more than one of the train, validation and test sets.

    .. code-block:: python

        from torcheeg.model_selection import KFoldCrossSubjectTrainValTest
        from torcheeg.datasets import DEAPDataset

        dataset = DEAPDataset(root_path='./data_preprocessed_python')
        cv = KFoldCrossSubjectTrainValTest(
            all_subjects=dataset.info['subject_id'].unique().tolist(), n_folds=5)
        for train_dataset, val_dataset, test_dataset in cv.split(dataset):
            ...

    Use at least three folds and at least one subject per fold. Subject IDs
    must be unique and match the dataset; filter the dataset first to evaluate
    a selected group of subjects. Metadata dtypes are preserved. Cached folds
    are reused only for the same configuration and ordered sample identities.
    Use a new :obj:`split_path` after changing the data or fold configuration.

    Args:
        all_subjects (list): The list of all subject ids to be partitioned into folds.
        n_folds (int): The number of folds. (default: :obj:`5`)
        split_path (str): The path to data partition information. If the path already exists, the existing partition is reused; otherwise, a new partition is created and saved there. (default: :obj:`None`)
        shuffle (bool): Whether to shuffle the subjects before splitting them into folds. (default: :obj:`True`)
        random_state (int): The random seed used to shuffle the subjects when :obj:`shuffle` is :obj:`True`. (default: :obj:`42`)
    '''

    def __init__(self,
                 all_subjects: List,
                 n_folds: int = 5,
                 split_path: Union[None, str] = None,
                 shuffle: bool = True,
                 random_state: int = 42):
        self.all_subjects = list(all_subjects)
        _validate_partition_ids([self.all_subjects])
        if (isinstance(n_folds, bool) or not isinstance(n_folds, Integral)
                or not 3 <= n_folds <= len(self.all_subjects)):
            raise ValueError('n_folds must be an integer between 3 and the number of subjects.')
        self.n_folds = int(n_folds)
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))
        self.shuffle = shuffle
        self.random_state = random_state

    def _make_folds(self):
        subjects = list(self.all_subjects)
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(subjects)
        return [subjects[index::self.n_folds] for index in range(self.n_folds)]

    def split_info_constructor(self, info: pd.DataFrame, split_path=None) -> None:
        if set(info['subject_id']) != set(self.all_subjects):
            raise ValueError('all_subjects must match the dataset subject_id values.')
        split_path = self.split_path if split_path is None else split_path
        folds = self._make_folds()
        for fold_id in range(self.n_folds):
            test_subjects = folds[fold_id]
            val_subjects = folds[(fold_id + 1) % self.n_folds]
            train_subjects = [subject for index, fold in enumerate(folds)
                              if index not in (fold_id, (fold_id + 1) % self.n_folds)
                              for subject in fold]
            _write_partition_indices(info, 'subject_id',
                                [train_subjects, val_subjects, test_subjects],
                                os.path.join(split_path, f'fold_{fold_id}'))

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = [f'fold_{fold_id}/{name}.csv' for fold_id in range(self.n_folds)
                 for name in ('train', 'val', 'test')]
        configuration = {'splitter': self.__class__.__name__,
                         'all_subjects': self.all_subjects, 'n_folds': self.n_folds,
                         'shuffle': self.shuffle, 'random_state': self.random_state}
        _ensure_split_cache(dataset.info, 'subject_id', self.split_path,
                           configuration, files,
                           lambda path: self.split_info_constructor(dataset.info, path))
        for fold_id in range(self.n_folds):
            yield tuple(_read_split_subset(dataset, self.split_path, f'fold_{fold_id}/{name}.csv')
                        for name in ('train', 'val', 'test'))
