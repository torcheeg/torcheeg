import os
import tempfile
import time
from contextlib import contextmanager
from typing import Union
from urllib.parse import quote, unquote

import numpy as np
import torch

from .base import _EEGSignalIO

_WRITE_TIMEOUT = 300.0


class MemmapEEGSignalIO(_EEGSignalIO):
    r'''
    Store EEG arrays as individual NumPy files and read them as read-only maps.

    Samples may have different shapes and non-object NumPy dtypes. Dense
    tensor inputs with NumPy-compatible dtypes are detached and copied to
    the CPU before conversion to NumPy.
    Object arrays are rejected without enabling pickle.

    The backend holds no open files between calls, so it can be copied or
    passed to spawned or forked data-loading workers. Each returned
    :obj:`numpy.memmap` owns its mapping until the array and its views are
    released; returning the array does not close its file handle. Keep that
    lifetime in mind when retaining many samples. Release maps before
    replacing an existing sample on platforms that restrict mapped-file
    replacement, including Windows.

    Writers use a per-store creation lock and publish complete files with an
    atomic rename. Readers do not take this lock. Explicit keys replace the
    corresponding sample; automatic integer keys never reuse an existing
    key. A lock left by a terminated writer produces a timeout. Remove it
    only after confirming that no writer remains active.

    Keys are encoded as filename components, so slashes, percent signs and
    Unicode do not escape the store or collide. Ordinary numeric keys from
    the earlier experimental backend remain compatible. Move experimental
    stores containing unencoded special-character keys to a fresh store.

    .. code-block:: python

        import numpy as np
        from torcheeg.io import EEGSignalIO

        eeg_io = EEGSignalIO('./signals', io_mode='memmap')
        key = eeg_io.write_eeg(np.ones((32, 128), dtype=np.float32))
        eeg = eeg_io.read_eeg(key)
        # eeg is a read-only numpy.memmap with shape (32, 128).
        writable_eeg = np.array(eeg, copy=True)

    Transforms that mutate samples in place must first make a writable copy.
    No conversion to a shared dense tensor or fixed sample shape is imposed.

    Args:
        io_path (str): Directory in which the per-sample NumPy files are stored.
    '''

    def __init__(self, io_path: str) -> None:
        self.io_path = os.path.abspath(os.fspath(io_path))
        os.makedirs(self.io_path, exist_ok=True)

    @staticmethod
    def _encode_key(key: str) -> str:
        if not isinstance(key, str) or not key:
            raise ValueError('Memmap keys must be nonempty strings.')
        return quote(key, safe='')

    def _sample_path(self, key: str) -> str:
        return os.path.join(self.io_path, self._encode_key(key) + '.npy')

    def __len__(self):
        return len(self.keys())

    def keys(self):
        r'''Return stored keys in deterministic filename order.'''
        keys = []
        for name in sorted(os.listdir(self.io_path)):
            if not name.endswith('.npy') or not os.path.isfile(os.path.join(self.io_path, name)):
                continue
            encoded = name[:-4]
            key = unquote(encoded, errors='strict')
            if self._encode_key(key) != encoded:
                raise ValueError('Noncanonical memmap filename; migrate this experimental store.')
            keys.append(key)
        return keys

    @property
    def write_pointer(self):
        numbers = [int(key) for key in self.keys() if key.isascii() and key.isdecimal()]
        return max(numbers, default=-1) + 1

    @contextmanager
    def _write_lock(self):
        lock_path = os.path.join(self.io_path, '.torcheeg-write.lock')
        deadline = time.monotonic() + _WRITE_TIMEOUT
        while True:
            try:
                descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Timeout waiting for memmap writer at {self.io_path}.')
                time.sleep(0.05)
        try:
            with os.fdopen(descriptor, 'w') as file:
                file.write(str(os.getpid()))
            yield
        finally:
            os.unlink(lock_path)

    def write_eeg(self, eeg: Union[any, torch.Tensor], key: Union[str, None] = None) -> str:
        r'''Atomically persist a NumPy-compatible array and return its key.'''
        if eeg is None:
            raise RuntimeError(f'Save None to the memmap with the key {key}!')
        if key is not None:
            self._encode_key(key)
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.detach().cpu().numpy()
        eeg = np.asarray(eeg)
        if eeg.dtype.hasobject:
            raise ValueError('Memmap EEG arrays must not contain Python objects.')
        with self._write_lock():
            if key is None:
                key = str(self.write_pointer)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix='.torcheeg-', suffix='.tmp', dir=self.io_path)
            try:
                with os.fdopen(descriptor, 'wb') as file:
                    np.save(file, eeg, allow_pickle=False)
                os.replace(temporary_path, self._sample_path(key))
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        return key

    def read_eeg(self, key: str) -> np.memmap:
        r'''Return a read-only map; the returned array owns the open mapping.'''
        path = self._sample_path(key)
        if not os.path.isfile(path):
            raise RuntimeError(f'Unable to index the EEG signal sample with key {key}!')
        return np.load(path, mmap_mode='r', allow_pickle=False)

    def eegs(self):
        r'''Return maps for all keys; each map remains open while referenced.'''
        return [self.read_eeg(key) for key in self.keys()]

    def __copy__(self):
        return self.__class__(self.io_path)
