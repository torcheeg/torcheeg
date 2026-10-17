import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from copy import copy

import numpy as np
import pandas as pd

log = logging.getLogger('torcheeg')
_CACHE_TIMEOUT = 300.0
_MANIFEST = '_split_manifest.json'


def validate_partition_ids(partitions):
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


def write_group_indices(info, column, partitions, split_path):
    r'''Write positions in the original row order without persisting metadata.'''
    validate_partition_ids(partitions)
    requested = set().union(*(set(ids) for ids in partitions))
    if requested - set(info[column]):
        raise ValueError(f'Unknown {column} values in the partition IDs.')
    os.makedirs(split_path, exist_ok=True)
    for name, ids in zip(('train', 'val', 'test'), partitions):
        positions = np.flatnonzero(info[column].isin(ids).to_numpy())
        pd.DataFrame({'index': positions}).to_csv(
            os.path.join(split_path, f'{name}.csv'), index=False)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _file_digest(path):
    with open(path, 'rb') as file:
        return _digest(file.read())


def _fingerprint(info, column):
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
    return _digest(json.dumps(header).encode() + hashes.tobytes())


def _verify_cache(split_path, expected, files):
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
            if _file_digest(os.path.join(split_path, name)) != checksums[name]:
                raise ValueError('Corrupt split cache; use a new split_path.')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Incomplete or corrupt split cache; use a new split_path.') from error


def ensure_split_cache(info, column, split_path, configuration, files, writer):
    r'''Publish a complete index cache under a process-safe creation lock.

    Existing caches are read-only. A failed writer removes only its own staging
    directory, allowing a later call to retry without reusing partial CSVs.
    '''
    columns = [column] if isinstance(column, str) else list(column)
    for name in columns:
        if name not in info or info[name].isna().any():
            raise ValueError(f'The dataset must contain nonmissing {name} values.')
    split_path = os.path.abspath(os.fspath(split_path))
    expected = {'version': 1, 'dataset': _fingerprint(info, column),
                'configuration': _digest(json.dumps(configuration, sort_keys=True,
                    default=lambda value: value.item() if isinstance(value, np.generic)
                    else str(value)).encode())}
    parent = os.path.dirname(split_path)
    os.makedirs(parent, exist_ok=True)
    lock_path = split_path + '.lock'
    deadline = time.monotonic() + _CACHE_TIMEOUT
    while True:
        if os.path.isfile(os.path.join(split_path, _MANIFEST)):
            _verify_cache(split_path, expected, files)
            log.info(f'📊 | Use existing split from {split_path}.')
            return
        if os.path.exists(split_path) and (not os.path.isdir(split_path)
                                          or os.listdir(split_path)):
            raise ValueError('Incomplete split cache; use a new split_path.')
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Timeout waiting for split creation at {split_path}.')
            time.sleep(0.05)
            continue
        staging_path = None
        try:
            with os.fdopen(descriptor, 'w') as file:
                file.write(str(os.getpid()))
            # A peer may publish the cache just before we acquire the lock.
            if os.path.isfile(os.path.join(split_path, _MANIFEST)):
                _verify_cache(split_path, expected, files)
                return
            staging_path = tempfile.mkdtemp(prefix='.torcheeg-split-', dir=parent)
            writer(staging_path)
            manifest = dict(expected)
            manifest['files'] = {name: _file_digest(os.path.join(staging_path, name))
                                 for name in files}
            with open(os.path.join(staging_path, _MANIFEST), 'w') as file:
                json.dump(manifest, file, sort_keys=True)
            if os.path.isdir(split_path):
                os.rmdir(split_path)
            os.replace(staging_path, split_path)
            log.info(f'📊 | Create split at {split_path}.')
            return
        finally:
            if staging_path is not None:
                shutil.rmtree(staging_path, ignore_errors=True)
            os.unlink(lock_path)


def read_index_subset(dataset, split_path, filename):
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
