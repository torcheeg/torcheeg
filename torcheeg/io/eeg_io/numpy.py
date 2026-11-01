import os
from urllib.parse import quote, unquote

import numpy as np

from ...utils import get_random_dir_path
from .base import BaseEEGIO


class NumpyEEGIO(BaseEEGIO):
    """Store numeric arrays as .npy files, optionally memory-mapped on read."""

    def __init__(self, io_path=None, mmap_mode='r'):
        if mmap_mode not in (None, 'r', 'r+', 'c'):
            raise ValueError("mmap_mode must be None, 'r', 'r+' or 'c'.")
        self.io_path = get_random_dir_path(
            dir_prefix='eeg') if io_path is None else os.fspath(io_path)
        self.mmap_mode = mmap_mode
        os.makedirs(self.io_path, exist_ok=True)

    def _filename(self, key, record=None):
        return os.path.join(self.io_path, quote(self.encode_key(key, record), safe='') + '.npy')

    def write_eeg(self, eeg, key, record=None):
        filename = self._filename(key, record)
        if eeg is None:
            raise ValueError('eeg must not be None.')
        array = np.asarray(eeg)
        if array.dtype.hasobject:
            raise TypeError('NumpyEEGIO requires arrays without object dtype.')
        np.save(filename, array, allow_pickle=False)
        return self.encode_key(key, record)

    def read_eeg(self, key, record=None):
        try:
            return np.load(self._filename(key, record), mmap_mode=self.mmap_mode, allow_pickle=False)
        except FileNotFoundError as error:
            raise KeyError(key) from error

    def keys(self):
        return [unquote(name[:-4]) for name in sorted(os.listdir(self.io_path)) if name.endswith('.npy')]

    def __len__(self):
        return len(self.keys())

    def clone(self, io_path):
        return type(self)(io_path, mmap_mode=self.mmap_mode)
