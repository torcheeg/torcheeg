import atexit
import logging
import os
import random
import shutil
import ssl
import string
import sys
import tempfile
import time
import urllib
from functools import partial
from typing import Optional

import hashlib
import json
from importlib import import_module as _import_module
from pathlib import Path

_log = logging.getLogger('torcheeg')


def get_temp_dir_path():
    temp_dir_path = tempfile.mkdtemp()
    atexit.register(partial(shutil.rmtree, temp_dir_path, ignore_errors=True))
    return temp_dir_path


def get_random_dir_path(root_path='.torcheeg', dir_prefix='tmp'):
    """
    Generates a unique folder name based on the current timestamp and a random suffix.
    The folder name is intended to be used for creating a temporary cache folder in torcheeg.

    Returns:
        str: A unique folder name.
    """
    timestamp = int(time.time() * 1000)

    random_suffix = ''.join(
        random.choices(string.ascii_letters + string.digits, k=5))

    dir_path = f"{dir_prefix}_{timestamp}_{random_suffix}"

    return os.path.join(root_path, dir_path)


def get_package_dir_path():
    home_dir = None

    if 'nt' == os.name.lower():
        if os.path.isdir(os.path.join(os.getenv('APPDATA'), '.torcheeg')):
            home_dir = os.getenv('APPDATA')
        else:
            home_dir = os.getenv('USERPROFILE')
    else:
        home_dir = os.path.expanduser('~')

    if home_dir is None:
        raise RuntimeError(
            'Cannot resolve home dictionary, please report this bug to TorchEEG Teams.'
        )

    return os.path.join(home_dir, '.torcheeg')


def download_url(url: str,
                 folder: str,
                 verbose: bool = True,
                 filename: Optional[str] = None):
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = os.path.join(folder, filename)

    if os.path.exists(path):  # pragma: no cover
        if verbose:
            _log.info(f'Using existing file {filename}', file=sys.stderr)
        return path

    if verbose:
        _log.info(f'Downloading {url}', file=sys.stderr)

    os.makedirs(os.path.expanduser(os.path.normpath(folder)))

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def compute_fingerprint(value):
    """Return the SHA256 digest of bytes describing cache inputs."""
    return hashlib.sha256(value).hexdigest()


def compute_checksum(path):
    """Return the SHA256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _list_files(path):
    return sorted(
        str(file.relative_to(path)) for file in Path(path).rglob('*') if file.is_file()
        and str(file.relative_to(path)) != 'manifest.json' and file.name != 'lock.mdb')


def validate_cache(path, fingerprint, files=None):
    """Raise ValueError if cache identity or output checksums do not match."""
    path = Path(path)
    try:
        with open(path / 'manifest.json', encoding='utf-8') as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict) or manifest.get('fingerprint') != fingerprint:
            raise ValueError(
                'Cache inputs or parameters have changed; use a new cache path.')
        checksums = manifest.get('files')
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError('Incomplete cache; use a new cache path.')
        if set(checksums) != set(_list_files(path)):
            raise ValueError('Cache output files have changed; use a new cache path.')
        if files is not None and set(checksums) != set(files):
            raise ValueError('Incomplete cache; use a new cache path.')
        for name, checksum in checksums.items():
            file = path / name
            if file.is_symlink() or os.path.commonpath(
                (str(file.resolve()), str(path.resolve()))) != str(path.resolve()):
                raise ValueError('Invalid cache output path.')
            if compute_checksum(file) != checksum:
                raise ValueError('Corrupt cache; use a new cache path.')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            'Incomplete or corrupt cache; use a new cache path.') from error


def ensure_cache(path, fingerprint, writer, files=None, validator=None):
    """Reuse a validated cache or build and publish it under a file lock."""
    from filelock import FileLock

    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with FileLock(path + '.lock', timeout=-1):
        if os.path.isfile(os.path.join(path, 'manifest.json')):
            validate_cache(path, fingerprint, files)
            if validator is not None:
                validator(path)
            return
        if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
            raise ValueError('Incomplete cache; use a new cache path.')
        staging_path = tempfile.mkdtemp(prefix='.torcheeg-cache-',
                                        dir=os.path.dirname(path))
        try:
            writer(staging_path)
            names = _list_files(staging_path)
            if not names or (files is not None and set(names) != set(files)):
                raise ValueError('Incomplete cache output.')
            if validator is not None:
                validator(staging_path)
            manifest = {
                'fingerprint': fingerprint,
                'files': {
                    name: compute_checksum(os.path.join(staging_path, name))
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


class LazyLoader:
    def __init__(self, lib_name, install_hint=None):
        self._lib_name = lib_name
        self._install_hint = install_hint
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = _import_module(self._lib_name)
            except ImportError as error:
                message = f"Could not import optional dependency `{self._lib_name}`."
                if self._install_hint:
                    message += f" {self._install_hint}"
                raise ImportError(message) from error
        return getattr(self._module, name)


__all__ = [
    'seed_everything', 'download_url', 'get_package_dir_path',
    'get_random_dir_path', 'get_temp_dir_path', 'LazyLoader',
    'compute_fingerprint', 'compute_checksum', 'validate_cache', 'ensure_cache',
]
