import hashlib
import json
import math
import os
import shutil
import tempfile
from numbers import Real
from pathlib import Path

from filelock import FileLock, Timeout

__all__ = [
    'get_cache_key', 'validate_cache', 'ensure_cache', 'validate_timeout',
    'get_file_checksum'
]


def validate_timeout(timeout):
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise TypeError('timeout must be a number of seconds.')
    if not math.isfinite(timeout) or (timeout < 0 and timeout != -1):
        raise ValueError('timeout must be nonnegative or -1.')
    return float(timeout)


def get_cache_key(value):
    """Return the SHA256 digest of bytes describing cache inputs."""
    return hashlib.sha256(value).hexdigest()


def get_file_checksum(path):
    """Return the SHA256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _get_files(path):
    return sorted(
        str(file.relative_to(path)) for file in Path(path).rglob('*') if file.is_file()
        and str(file.relative_to(path)) != 'manifest.json' and file.name != 'lock.mdb')


def validate_cache(path, key, files=None):
    """Raise ValueError if cache identity or output checksums do not match."""
    path = Path(path)
    try:
        with open(path / 'manifest.json', encoding='utf-8') as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict) or manifest.get('key') != key:
            raise ValueError(
                'Cache inputs or parameters have changed; use a new cache path.')
        checksums = manifest.get('files')
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError('Incomplete cache; use a new cache path.')
        if set(checksums) != set(_get_files(path)):
            raise ValueError('Cache output files have changed; use a new cache path.')
        if files is not None and set(checksums) != set(files):
            raise ValueError('Incomplete cache; use a new cache path.')
        for name, checksum in checksums.items():
            file = path / name
            if file.is_symlink() or os.path.commonpath(
                (str(file.resolve()), str(path.resolve()))) != str(path.resolve()):
                raise ValueError('Invalid cache output path.')
            if get_file_checksum(file) != checksum:
                raise ValueError('Corrupt cache; use a new cache path.')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            'Incomplete or corrupt cache; use a new cache path.') from error


def ensure_cache(path, key, writer, files=None, timeout=300.0, validator=None):
    """Reuse a validated cache or build and publish it under a file lock."""
    timeout = validate_timeout(timeout)
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with FileLock(path + '.lock', timeout=timeout):
            if os.path.isfile(os.path.join(path, 'manifest.json')):
                validate_cache(path, key, files)
                if validator is not None:
                    validator(path)
                return
            if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
                raise ValueError('Incomplete cache; use a new cache path.')
            staging_path = tempfile.mkdtemp(prefix='.torcheeg-cache-',
                                            dir=os.path.dirname(path))
            try:
                writer(staging_path)
                names = _get_files(staging_path)
                if not names or (files is not None and set(names) != set(files)):
                    raise ValueError('Incomplete cache output.')
                if validator is not None:
                    validator(staging_path)
                manifest = {
                    'key': key,
                    'files': {
                        name: get_file_checksum(os.path.join(staging_path, name))
                        for name in names
                    }
                }
                with open(os.path.join(staging_path, 'manifest.json'),
                          'w',
                          encoding='utf-8') as file:
                    json.dump(manifest, file, sort_keys=True)
                if os.path.isdir(path):
                    os.rmdir(path)
                os.replace(staging_path, path)
            finally:
                shutil.rmtree(staging_path, ignore_errors=True)
    except Timeout as error:
        raise RuntimeError(f'Timeout waiting for cache creation at {path}.') from error
