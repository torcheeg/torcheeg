import os
import pickle
import threading
import weakref
from typing import Union

import lmdb
import torch

from ...utils import get_random_dir_path
from .base import BaseEEGIO

# Weak values let the last owner release the native environment.
_LMDB_ENV_CACHE = weakref.WeakValueDictionary()
_LMDB_ENV_LOCK = threading.Lock()
_LMDB_ENV_PID = os.getpid()


def _reset_lmdb_after_fork() -> None:
    global _LMDB_ENV_CACHE, _LMDB_ENV_LOCK, _LMDB_ENV_PID
    inherited = list(_LMDB_ENV_CACHE.values())
    _LMDB_ENV_CACHE = weakref.WeakValueDictionary()
    # An inherited lock may have been held by a thread absent in the child.
    _LMDB_ENV_LOCK = threading.Lock()
    _LMDB_ENV_PID = os.getpid()
    for environment in inherited:
        environment.close()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_lmdb_after_fork)


def _get_lmdb_environment(path: str, io_size: int) -> lmdb.Environment:
    if _LMDB_ENV_PID != os.getpid():
        _reset_lmdb_after_fork()
    with _LMDB_ENV_LOCK:
        environment = _LMDB_ENV_CACHE.get(path)
        if environment is not None:
            try:
                environment.info()
            except lmdb.Error:
                environment = None
        if environment is None:
            environment = lmdb.open(path=path, map_size=io_size, lock=False,
                                    max_spare_txns=0)
            _LMDB_ENV_CACHE[path] = environment
        return environment


class LMDBEEGIO(BaseEEGIO):

    def __init__(self, io_path: str = None, io_size: int = 1048576) -> None:
        self.io_path = get_random_dir_path(
            dir_prefix="eeg") if io_path is None else os.fspath(io_path)
        self.io_size = io_size

        os.makedirs(self.io_path, exist_ok=True)
        self._env_path = os.path.realpath(self.io_path)
        self._env_obj = None
        self._env_pid = None

    @property
    def _env(self) -> lmdb.Environment:
        if self._env_obj is None or self._env_pid != os.getpid():
            self._env_obj = _get_lmdb_environment(self._env_path, self.io_size)
            self._env_pid = os.getpid()
        return self._env_obj

    def __len__(self):
        with self._env.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str, record=None) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): Sample key. An existing key replaces its sample.

        Returns:
            str: The key of the written EEG signal.
        '''
        key = self.encode_key(key, record)

        if not isinstance(key, str):
            raise TypeError('key must be a string.')

        if eeg is None:
            raise ValueError(f'Save None to the LMDB with the key {key}!')

        try_again = False
        try:
            with self._env.begin(write=True) as transaction:
                transaction.put(key.encode(), pickle.dumps(eeg))
        except lmdb.MapFullError:
            self.io_size = max(self.io_size, self._env.info()['map_size']) * 2
            self._env.set_mapsize(self.io_size)
            try_again = True
        if try_again:
            return self.write_eeg(key=key, eeg=eeg)
        return key

    def read_eeg(self, key: str, record=None) -> any:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        key = self.encode_key(key, record)
        with self._env.begin(write=False) as transaction:
            eeg = transaction.get(key.encode())

        if eeg is None:
            raise KeyError(
                f'Unable to index the EEG signal sample with key {key}!')

        return pickle.loads(eeg)

    def keys(self):
        r'''
        Get all keys in the LMDBEEGIO.

        Returns:
            list: The list of keys in the LMDBEEGIO.
        '''
        with self._env.begin(write=False) as transaction:
            return [
                key.decode()
                for key in transaction.cursor().iternext(keys=True,
                                                         values=False)
            ]

    def eegs(self):
        r'''
        Get all EEG signals in the LMDBEEGIO.

        Returns:
            list: The list of EEG signals in the LMDBEEGIO.
        '''
        return [self.read_eeg(key) for key in self.keys()]

    def __getstate__(self):
        # Each process must acquire its own environment.
        state = self.__dict__.copy()
        state['_env_obj'] = None
        state['_env_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env_path = state.get('_env_path', os.path.realpath(self.io_path))
        self._env_obj = None
        self._env_pid = None

    def __copy__(self):
        # LMDB does not allow reopening the same environment in one process.
        return self

    def close(self):
        self._env_obj = None
        self._env_pid = None

    def clone(self, io_path):
        return type(self)(io_path=io_path, io_size=self.io_size)
