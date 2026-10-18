import hashlib
import json
import math
from numbers import Real
import os
import shutil
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

_CACHE_VERSION = 2
_MANIFEST = 'manifest.json'


def validate_timeout(timeout):
    if isinstance(timeout, (bool, np.bool_)) or not isinstance(timeout, Real):
        raise TypeError('timeout must be a number of seconds.')
    if not math.isfinite(timeout) or (timeout < 0 and timeout != -1):
        raise ValueError('timeout must be nonnegative or -1.')
    return float(timeout)


def validate_metadata(info, columns=()):
    if not isinstance(info, pd.DataFrame):
        raise TypeError('Dataset metadata must be a pandas DataFrame.')
    if info.empty:
        raise ValueError('The dataset must contain nonempty metadata.')
    columns = [columns] if isinstance(columns, str) else list(columns)
    for column in columns:
        if column not in info or info[column].isna().any():
            raise ValueError(f'The dataset must contain nonmissing {column} values.')


def validate_group_names(values):
    names = [str(value) for value in values]
    if len(set(names)) != len(names):
        raise ValueError('Subject IDs must have distinct string representations.')
    return names


def validate_partition_ids(partitions):
    r'''Reject empty, repeated or overlapping train/validation/test IDs.'''
    seen = set()
    for ids in partitions:
        if len(ids) == 0 or pd.Series(ids, dtype=object).isna().any():
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


def write_partition_indices(info, column, partitions, split_path):
    r'''Write positions in the original row order without persisting metadata.'''
    if len(partitions) != 3:
        raise ValueError('Expected train, validation and test partitions.')
    validate_partition_ids(partitions)
    requested = set().union(*(set(ids) for ids in partitions))
    if requested - set(info[column]):
        raise ValueError(f'Unknown {column} values in the partition IDs.')
    os.makedirs(split_path, exist_ok=True)
    for name, ids in zip(('train', 'val', 'test'), partitions):
        positions = np.flatnonzero(info[column].isin(ids).to_numpy())
        write_indices(os.path.join(split_path, f'{name}.csv'), positions)


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_file(path):
    with open(path, 'rb') as file:
        return _hash_bytes(file.read())


def _fingerprint_dataset(info, column):
    columns = [column] if isinstance(column, str) else list(column)
    if not any(name in info for name in ('_record_id', 'clip_id')):
        columns = list(info.columns)
    columns += [
        name for name in ('_record_id', 'clip_id')
        if name in info and name not in columns
    ]
    header = [(name, str(info[name].dtype)) for name in columns]
    typed_values = pd.DataFrame({
        name: [(type(value).__name__, value) for value in info[name].tolist()]
        for name in columns
    })
    hashes = pd.util.hash_pandas_object(typed_values, index=False).to_numpy()
    return _hash_bytes(json.dumps(header).encode() + hashes.tobytes())


def _validate_split_cache(split_path, expected, files):
    try:
        with open(os.path.join(split_path, _MANIFEST)) as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError('Corrupt split cache; use a new split_path.')
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                'Split parameters or dataset order have changed; use a new split_path.')
        checksums = manifest['files']
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError('Incomplete split cache; use a new split_path.')
        if files is not None and set(checksums) != set(files):
            raise ValueError('Incomplete split cache; use a new split_path.')
        if files is None:
            files = list(checksums)
            actual = {
                str(path.relative_to(split_path))
                for path in Path(split_path).rglob('*.csv')
            }
            if actual != set(files):
                raise ValueError('Incomplete split cache; use a new split_path.')
        for name in files:
            if _hash_file(os.path.join(split_path, name)) != checksums[name]:
                raise ValueError('Corrupt split cache; use a new split_path.')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            'Incomplete or corrupt split cache; use a new split_path.') from error


def _serialize_configuration(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.random.RandomState):
        return value.get_state()
    raise TypeError(f'Unsupported split configuration value: {type(value).__name__}')


def ensure_split_cache(info,
                       columns,
                       split_path,
                       configuration,
                       files,
                       writer,
                       timeout=300.0):
    r'''Create or reuse a complete split cache under a process-safe lock.'''
    timeout = validate_timeout(timeout)
    validate_metadata(info, columns)
    split_path = os.path.abspath(os.fspath(split_path))
    configuration_hash = _hash_bytes(
        json.dumps(configuration, sort_keys=True,
                   default=_serialize_configuration).encode())
    expected = {
        'version': _CACHE_VERSION,
        'format': 'row_indices',
        'dataset': _fingerprint_dataset(info, columns),
        'configuration': configuration_hash
    }
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    try:
        with FileLock(split_path + '.lock', timeout=timeout):
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
                names = files if files is not None else sorted(
                    str(path.relative_to(staging_path))
                    for path in Path(staging_path).rglob('*.csv'))
                if not names:
                    raise ValueError('The split cache must contain at least one file.')
                for name in names:
                    if name.endswith('.csv'):
                        read_indices(os.path.join(staging_path, name), len(info))
                manifest['files'] = {
                    name: _hash_file(os.path.join(staging_path, name))
                    for name in names
                }
                with open(os.path.join(staging_path, _MANIFEST), 'w') as file:
                    json.dump(manifest, file, sort_keys=True)
                if os.path.isdir(split_path):
                    os.rmdir(split_path)
                os.replace(staging_path, split_path)
            finally:
                shutil.rmtree(staging_path, ignore_errors=True)
    except Timeout as error:
        raise RuntimeError(
            f'Timeout waiting for split creation at {split_path}.') from error


def validate_indices(indices, num_samples):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or indices.size == 0
            or (indices < 0).any() or (indices >= num_samples).any()
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Split cache contains invalid row positions.')
    return indices


def write_indices(path, indices):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not indices.size
            or (indices < 0).any() or len(np.unique(indices)) != len(indices)):
        raise ValueError('Split indices must be nonempty integer row positions.')
    os.makedirs(os.path.dirname(os.fspath(path)) or '.', exist_ok=True)
    pd.DataFrame({'index': indices}).to_csv(path, index=False)


def read_indices(path, num_samples):
    frame = pd.read_csv(path)
    if list(frame.columns) != ['index'] or frame['index'].dtype.kind not in 'iu':
        raise ValueError('Split cache must contain integer row positions.')
    return validate_indices(frame['index'].to_numpy(), num_samples)


def create_subset(dataset, indices):
    indices = validate_indices(indices, len(dataset))
    subset = copy(dataset)
    subset.info = dataset.info.iloc[indices].reset_index(drop=True)
    return subset


def read_subset(dataset, split_path, filename):
    return create_subset(dataset,
                         read_indices(os.path.join(split_path, filename), len(dataset)))
