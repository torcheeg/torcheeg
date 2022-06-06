import os
from typing import Tuple, Union

import lmdb
import numpy as np

MAX_LMDB_SIZE = 1099511627776


class EEGSignalIO:
    r'''
    A general-purpose, lightweight and efficient EEG signal IO APIs for converting various real-world EEG signal datasets into :obj:`np.ndarray` samples and storing them in the database. Here, we draw on the implementation ideas of industrial-grade application Caffe, and encapsulate a set of EEG signal reading and writing methods based on Lightning Memory-Mapped Database (LMDB), which not only unifies the differences of data types in different databases, but also accelerates the reading of data during training and testing.

    .. code-block:: python

        eeg_io = EEGSignalIO('YOUR_PATH')
        key = eeg_io.write_eeg(np.random.randn(32, 128))
        eeg = eeg_io.read_eeg(key)
        eeg.shape
        >>> (32, 128)
    
    Args:
        cache_path (str): Where the database is stored.
        cache_size (int): The maximum capacity of the database. (default: :obj:`1099511627776`)
    '''
    def __init__(self,
                 cache_path: str,
                 cache_size: int = MAX_LMDB_SIZE) -> None:
        self.cache_path = cache_path
        self.cache_size = cache_size

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

        self.db_eeg_size = None
        self.db_eeg_dtype = None

    @property
    def write_pointer(self):
        return len(self)

    def __len__(self):
        with lmdb.open(self.cache_path, self.cache_size, lock=False) as env:
            with env.begin() as transaction:
                # number of samples after removing db_eeg_size and db_eeg_dtype
                length = transaction.stat()['entries'] - 2
                return length if length > 0 else 0

    def set_eeg_format(self, eeg: np.ndarray) -> None:
        r'''
        Set the sample data shapes and types acceptable to the database based on the inserted EEG signals.

        Args:
            eeg (np.ndarray): The EEG signal sample to be written into the database.
        '''
        self.db_eeg_dtype = eeg.dtype
        self.db_eeg_size = eeg.shape

        with lmdb.open(self.cache_path, self.cache_size, lock=False) as env:
            with env.begin(write=True) as transaction:
                transaction.put('db_eeg_size'.encode(),
                                np.array(eeg.shape, dtype=np.uint32).tobytes())
                transaction.put('db_eeg_dtype'.encode(),
                                str(eeg.dtype).encode())

    def get_eeg_format(self) -> Tuple[np.ndarray, str]:
        r'''
        Get the sample data shapes and types acceptable to the database.

        Returns:
            tuple(np.ndarray, str): the sample data shapes and types acceptable to the database.
        '''
        with lmdb.open(self.cache_path, self.cache_size, lock=False) as env:
            with env.begin() as transaction:
                db_eeg_size = transaction.get('db_eeg_size'.encode())
                db_eeg_dtype = transaction.get('db_eeg_dtype'.encode())
            if (db_eeg_size is None) or (db_eeg_dtype is None):
                return (None, None)
            return tuple(np.frombuffer(
                db_eeg_size, dtype=np.uint32).tolist()), db_eeg_dtype.decode()

    def write_eeg(self, eeg: np.ndarray, key: Union[str, None] = None) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (np.ndarray): EEG signal samples to be written into the database.
            key (str, optional): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing integer.

        Returns:
            int: The index of written EEG signals in the database.
        '''
        if (self.db_eeg_size is None) or (self.db_eeg_dtype is None):
            self.db_eeg_size, self.db_eeg_dtype = self.get_eeg_format()
            if (self.db_eeg_size is None) or (self.db_eeg_dtype is None):
                self.set_eeg_format(eeg)
        elif self.db_eeg_size != eeg.shape:
            raise RuntimeError(
                f'Inserting eeg sample failed. The dimensions of the eeg array in the database, {self.db_eeg_size}, do not match the sample to be inserted, {eeg.shape}.'
            )
        elif self.db_eeg_dtype != eeg.dtype:
            raise RuntimeError(
                f'Inserting eeg sample failed. The dtype of the eeg array in the database, {self.db_eeg_dtype}, do not match the sample to be inserted, {eeg.dtype}.'
            )

        if key is None:
            key = str(self.write_pointer)

        with lmdb.open(self.cache_path, self.cache_size, lock=False) as env:
            with env.begin(write=True) as transaction:
                transaction.put(key.encode(), eeg.tobytes())
            return key

    def read_eeg(self, key: str) -> np.ndarray:
        r'''
        Query the corresponding EEG signal in the database according to the index.

        Args:
            key (str): The index of the EEG signal to be queried.
            
        Returns:
            np.ndarray: The EEG signal sample.
        '''
        if (self.db_eeg_size is None) or (self.db_eeg_dtype is None):
            self.db_eeg_size, self.db_eeg_dtype = self.get_eeg_format()

        with lmdb.open(self.cache_path, self.cache_size, lock=False) as env:
            with env.begin() as transaction:
                eeg = transaction.get(key.encode())
            if eeg is None:
                raise RuntimeError(f'Unable to index the EEG signal sample with key {key}!')
            return np.frombuffer(eeg, dtype=self.db_eeg_dtype).reshape(
                self.db_eeg_size)