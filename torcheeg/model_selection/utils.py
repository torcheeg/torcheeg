import hashlib
import json
import os
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from ..utils.cache import (ensure_cache as _ensure_cache, get_cache_key as
                           _get_cache_key, validate_cache, validate_timeout)

__all__ = [
    'get_cache_key', 'validate_cache', 'ensure_cache', 'validate_timeout',
    'validate_metadata', 'validate_group_names', 'validate_partition_ids',
    'write_partition_indices', 'validate_indices', 'write_indices', 'read_indices',
    'create_subset'
]


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


def _serialize_configuration(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.random.RandomState):
        return value.get_state()
    raise TypeError(f'Unsupported split configuration value: {type(value).__name__}')


def get_cache_key(info, columns, configuration):
    """Return a key for ordered sample identities and split configuration."""
    validate_metadata(info, columns)
    return _get_cache_key(
        json.dumps(
            {
                'dataset': _fingerprint_dataset(info, columns),
                'configuration': configuration
            },
            sort_keys=True,
            default=_serialize_configuration).encode())


def ensure_cache(info,
                 columns,
                 split_path,
                 configuration,
                 files,
                 writer,
                 timeout=300.0):
    """Create or reuse validated row-position files."""
    key = get_cache_key(info, columns, configuration)

    def validate_outputs(path):
        names = files if files is not None else [
            str(file.relative_to(path)) for file in Path(path).rglob('*.csv')
        ]
        for name in names:
            read_indices(os.path.join(path, name), len(info))

    _ensure_cache(split_path,
                  key,
                  writer,
                  files=files,
                  timeout=timeout,
                  validator=validate_outputs)


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
