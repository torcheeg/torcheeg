import hashlib
import json
import os
import shutil
import tempfile
from copy import copy
from typing import Dict, Iterator, Tuple, Union

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

class TrainValTestSplitGroupbyFolder:
    r'''
    A tool class for datasets where the train/validation/test split is
    determined by the source folder (e.g., ``Training set/``,
    ``Validation set/``, ``Test set/``). The split is read from a ``split``
    column in the dataset's info DataFrame, whose values are ``'train'``,
    ``'val'``, and ``'test'``.

    .. code-block:: python

        from torcheeg.model_selection import TrainValTestSplitGroupbyFolder

        splitter = TrainValTestSplitGroupbyFolder()
        for train_dataset, val_dataset, test_dataset in splitter.split(dataset):
            ...

    Each row must have one of the three split values, and all three subsets
    must be nonempty. Folder labels determine the split; this does not itself
    guarantee that subjects or trials are disjoint across the source folders.
    Metadata dtypes are preserved. Existing caches must match the configuration
    and ordered sample IDs; use a new :obj:`split_path` after changing the data.

    Args:
        split_path (str): The path to data partition information. If the path
            exists, read the existing partition from the path. If the path does
            not exist, the current division method will be saved for next use.
            If set to None, a random path will be generated.
            (default: :obj:`None`)
    '''

    def __init__(self, split_path: Union[None, str] = None):
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))

    def split_info_constructor(self, info: pd.DataFrame, split_path=None) -> None:
        if not info['split'].isin(['train', 'val', 'test']).all():
            raise ValueError("The split column must contain only 'train', 'val' and 'test'.")
        _write_partition_indices(info, 'split', [['train'], ['val'], ['test']],
                            self.split_path if split_path is None else split_path)

    def split(self, dataset: BaseDataset) -> Iterator[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        files = ['train.csv', 'val.csv', 'test.csv']
        _ensure_split_cache(dataset.info, 'split', self.split_path,
                           {'splitter': self.__class__.__name__}, files,
                           lambda path: self.split_info_constructor(dataset.info, path))
        yield tuple(_read_split_subset(dataset, self.split_path, name) for name in files)

    @property
    def repr_body(self) -> Dict:
        return {'split_path': self.split_path}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split_path={self.split_path!r})"
