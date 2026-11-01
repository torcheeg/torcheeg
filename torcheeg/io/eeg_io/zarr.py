import os
from urllib.parse import quote, unquote

import numpy as np

from ...utils import LazyLoader, get_random_dir_path
from .base import BaseEEGIO

_zarr = LazyLoader('zarr', install_hint='Install with `pip install zarr`.')


class ZarrEEGIO(BaseEEGIO):
    """Store numeric arrays in a Zarr group, reading each sample on demand."""

    def __init__(self, io_path=None):
        self.io_path = get_random_dir_path(
            dir_prefix='eeg') if io_path is None else os.fspath(io_path)
        self._group('a')

    def _group(self, mode):
        return _zarr.open_group(self.io_path, mode=mode)

    def write_eeg(self, eeg, key, record=None):
        encoded = self.encode_key(key, record)
        name = '%2E' * len(encoded) if encoded in ('.',
                                                   '..') else quote(encoded, safe='')
        if eeg is None:
            raise ValueError('eeg must not be None.')
        array = np.asarray(eeg)
        if array.dtype.hasobject:
            raise TypeError('ZarrEEGIO requires arrays without object dtype.')
        group = self._group('a')
        if hasattr(group, 'create_array'):
            group.create_array(name, data=array, overwrite=True)
        else:
            group.create_dataset(name, data=array, overwrite=True)
        return self.encode_key(key, record)

    def read_eeg(self, key, record=None):
        encoded = self.encode_key(key, record)
        name = '%2E' * len(encoded) if encoded in ('.',
                                                   '..') else quote(encoded, safe='')
        try:
            return np.asarray(self._group('r')[name])
        except (KeyError, FileNotFoundError) as error:
            raise KeyError(key) from error

    def keys(self):
        if not os.path.exists(self.io_path):
            return []
        return [unquote(name) for name in sorted(self._group('r').array_keys())]

    def __len__(self):
        return len(self.keys())

    def clone(self, io_path):
        return type(self)(io_path)
