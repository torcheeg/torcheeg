import os
import pickle
import threading
import weakref
from typing import Union

import lmdb
import torch

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


class _EEGSignalIO:

    def keys(self):
        raise NotImplementedError(
            "Method keys is not implemented in class "
            + self.__class__.__name__
        )

    def eegs(self):
        raise NotImplementedError(
            "Method eegs is not implemented in class "
            + self.__class__.__name__
        )

    def read_eeg(self, key: str) -> any:
        raise NotImplementedError(
            "Method read_eeg is not implemented in class "
            + self.__class__.__name__
        )

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str) -> str:
        raise NotImplementedError(
            "Method write_eeg is not implemented in class "
            + self.__class__.__name__
        )


class MemoryEEGSignalIO(_EEGSignalIO):

    def __init__(self):
        self._memory = {}

    def __len__(self):
        return len(self._memory)

    def keys(self):
        r'''
        Get all keys in the MemoryEEGSignalIO.

        Returns:
            list: The list of keys in the MemoryEEGSignalIO.
        '''
        return list(self._memory.keys())

    def eegs(self):
        return list(self._memory.values())

    def read_eeg(self, key: str) -> any:
        r'''
        Read all the MemoryEEGSignalIO into memory, and index the specified EEG signal in memory with the given :obj:`key`.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        if key not in self._memory:
            raise KeyError(
                f'Unable to index the EEG signal sample with key {key}!')

        return self._memory[key]

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str) -> str:
        r'''
        Write EEG signal to memory.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): Sample key. An existing key replaces its sample.
        '''
        if not isinstance(key, str):
            raise TypeError('key must be a string.')

        if eeg is None:
            raise ValueError(f'Save None to the memory with the key {key}!')

        self._memory[key] = eeg

        return key

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class LMDBEEGSignalIO(_EEGSignalIO):

    def __init__(self, io_path: str, io_size: int = 1048576) -> None:
        self.io_path = io_path
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
                  key: str) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): Sample key. An existing key replaces its sample.

        Returns:
            str: The key of the written EEG signal.
        '''

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

    def read_eeg(self, key: str) -> any:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        with self._env.begin(write=False) as transaction:
            eeg = transaction.get(key.encode())

        if eeg is None:
            raise KeyError(
                f'Unable to index the EEG signal sample with key {key}!')

        return pickle.loads(eeg)

    def keys(self):
        r'''
        Get all keys in the LMDBEEGSignalIO.

        Returns:
            list: The list of keys in the LMDBEEGSignalIO.
        '''
        with self._env.begin(write=False) as transaction:
            return [
                key.decode()
                for key in transaction.cursor().iternext(keys=True,
                                                         values=False)
            ]

    def eegs(self):
        r'''
        Get all EEG signals in the LMDBEEGSignalIO.

        Returns:
            list: The list of EEG signals in the LMDBEEGSignalIO.
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


class PickleEEGSignalIO(_EEGSignalIO):

    def __init__(self, io_path: str) -> None:
        self.io_path = io_path

        os.makedirs(self.io_path, exist_ok=True)

    def __len__(self):
        return len(os.listdir(self.io_path))

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str) -> str:
        r'''
            Write EEG signal to folder.

            Args:
                eeg (any): EEG signal samples to be written into the folder.
                key (str): Sample key. An existing key replaces its sample.

            Returns:
                str: The key of the written EEG signal.
            '''

        if not isinstance(key, str):
            raise TypeError('key must be a string.')

        if eeg is None:
            raise ValueError(f'Save None to the LMDB with the key {key}!')

        with open(os.path.join(self.io_path, key), 'wb') as f:
            pickle.dump(eeg, f)

        return key

    def read_eeg(self, key: str) -> any:
        r'''
            Query the corresponding EEG signal in the folder according to the index.

            Args:
                key (str): The index of the EEG signal to be queried.

            Returns:
                any: The EEG signal sample.
            '''
        try:
            with open(os.path.join(self.io_path, key), 'rb') as f:
                eeg = pickle.load(f)
        except FileNotFoundError as error:
            raise KeyError(key) from error

        return eeg

    def keys(self):
        r'''
            Get all keys in the PickleEEGSignalIO.

            Returns:
                list: The list of keys in the PickleEEGSignalIO.
            '''
        return os.listdir(self.io_path)

    def eegs(self):
        r'''
            Get all EEG signals in the PickleEEGSignalIO.

            Returns:
                list: The list of EEG signals in the PickleEEGSignalIO.
            '''
        return [self.read_eeg(key) for key in self.keys()]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class EEGSignalIO:
    r'''
    A general-purpose, lightweight and efficient EEG signal IO APIs for converting various real-world EEG signal datasets into samples and storing them in the database. Here, we draw on the implementation ideas of industrial-grade application Caffe, and encapsulate a set of EEG signal reading and writing methods based on Lightning Memory-Mapped Database (LMDB), which not only unifies the differences of data types in different databases, but also accelerates the reading of data during training and testing.

    .. code-block:: python

        eeg_io = EEGSignalIO('YOUR_PATH')
        key = eeg_io.write_eeg(np.random.randn(32, 128), key='sample_0')
        eeg = eeg_io.read_eeg(key)
        eeg.shape
        >>> (32, 128)

    Args:
        io_path (str): Where the database is stored.
        io_size (int, optional): The maximum capacity of the database. It will increase according to the size of the dataset. (default: :obj:`1024`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems. Here, a file system based and a memory based EEG signal storages are also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
    '''

    def __init__(self,
                 io_path: str,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb') -> None:
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode

        if self.io_mode == 'lmdb':
            self._io = LMDBEEGSignalIO(io_path=self.io_path,
                                       io_size=self.io_size)
        elif self.io_mode == 'pickle':
            self._io = PickleEEGSignalIO(io_path=self.io_path)
        elif self.io_mode == 'memory':
            self._io = MemoryEEGSignalIO()
        else:
            raise ValueError(
                f'Unsupported io_mode {self.io_mode}, please choose from lmdb, pickle and memory.'
            )

    def __del__(self):
        if hasattr(self, '_io'):
            del self._io

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != '_io'
        })
        result._io = self._io.__copy__()
        return result

    def __len__(self):
        return len(self._io)

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): Sample key. An existing key replaces its sample.

        Returns:
            str: The key of the written EEG signal.
        '''

        return self._io.write_eeg(eeg=eeg, key=key)

    def read_eeg(self, key: str) -> any:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        return self._io.read_eeg(key)

    def keys(self):
        r'''
        Get all keys in the EEGSignalIO.

        Returns:
            list: The list of keys in the EEGSignalIO.
        '''
        return self._io.keys()

    def eegs(self):
        r'''
        Get all EEG signals in the EEGSignalIO.

        Returns:
            list: The list of EEG signals in the EEGSignalIO.
        '''
        return self._io.eegs()

    def to_lmdb(self, io_path: str, io_size: int = 1048576):
        r'''
        Convert to the LMDBEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = LMDBEEGSignalIO(io_path=io_path, io_size=io_size)

        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = 'lmdb'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io

    def to_pickle(self, io_path: str):
        r'''
        Convert to the PickleEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = PickleEEGSignalIO(io_path=io_path)

        self.io_path = io_path
        self.io_mode = 'pickle'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io

    def to_memory(self):
        r'''
        Convert to the MemoryEEGSignalIO, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.
        '''
        _io = MemoryEEGSignalIO()

        self.io_mode = 'memory'

        for key in self.keys():
            _io.write_eeg(self.read_eeg(key=key), key=key)

        self._io = _io
