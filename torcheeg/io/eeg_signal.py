import pickle
import os
from typing import Union

import torch
import lmdb


class EEGSignalIO:
    r'''
    A general-purpose, lightweight and efficient EEG signal IO APIs for converting various real-world EEG signal datasets into samples and storing them in the database. Here, we draw on the implementation ideas of industrial-grade application Caffe, and encapsulate a set of EEG signal reading and writing methods based on Lightning Memory-Mapped Database (LMDB), which not only unifies the differences of data types in different databases, but also accelerates the reading of data during training and testing.

    .. code-block:: python

        eeg_io = EEGSignalIO('YOUR_PATH')
        key = eeg_io.write_eeg(np.random.randn(32, 128))
        eeg = eeg_io.read_eeg(key)
        eeg.shape
        >>> (32, 128)
    
    Args:
        io_path (str): Where the database is stored.
        io_size (int, optional): The maximum capacity of the database. It will increase according to the size of the dataset. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
    '''
    def __init__(self,
                 io_path: str,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb') -> None:
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode

        assert io_mode in [
            'lmdb', 'pickle'
        ], f'Unsupported io_mode {io_mode}, please choose from [lmdb, pickle]!'

        self._in_memory = None

        if not os.path.exists(self.io_path):
            os.makedirs(self.io_path, exist_ok=True)

    @property
    def write_pointer(self):
        return len(self)

    def __len__(self):
        if self.io_mode == 'pickle':
            return len(os.listdir(self.io_path))

        if self.io_mode == 'lmdb':
            with lmdb.open(path=self.io_path,
                           map_size=self.io_size,
                           lock=False) as env:
                with env.begin() as transaction:
                    return transaction.stat()['entries']

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: Union[str, None] = None) -> str:
        r'''
        Write EEG signal to database.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str, optional): The key of the EEG signal to be inserted, if not specified, it will be an auto-incrementing integer.

        Returns:
            int: The index of written EEG signals in the database.
        '''

        if key is None:
            key = str(self.write_pointer)

        if eeg is None:
            raise RuntimeError(f'Save None to the LMDB with the key {key}!')

        if self.io_mode == 'pickle':
            with open(os.path.join(self.io_path, key), 'wb') as f:
                pickle.dump(eeg, f)
            return key

        if self.io_mode == 'lmdb':
            try_again = False
            with lmdb.open(path=self.io_path,
                           map_size=self.io_size,
                           lock=False,
                           writemap=True,
                           map_async=True) as env:
                try:
                    with env.begin(write=True) as transaction:
                        transaction.put(key.encode(), pickle.dumps(eeg))
                except lmdb.MapFullError:
                    # print(
                    #     f'The current io_size is not enough, and double the LMDB map size to {self.io_size * 2} automatically.'
                    # )
                    self.io_size = self.io_size * 2
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
        if self.io_mode == 'pickle':
            with open(os.path.join(self.io_path, key), 'rb') as f:
                eeg = pickle.load(f)
            return eeg

        if self.io_mode == 'lmdb':
            with lmdb.open(path=self.io_path,
                           map_size=self.io_size,
                           lock=False) as env:
                with env.begin() as transaction:
                    eeg = transaction.get(key.encode())

                if eeg is None:
                    raise RuntimeError(
                        f'Unable to index the EEG signal sample with key {key}!'
                    )

                return pickle.loads(eeg)

    def keys(self):
        r'''
        Get all keys in the EEGSignalIO.

        Returns:
            list: The list of keys in the EEGSignalIO.
        '''
        if self.io_mode == 'pickle':
            return os.listdir(self.io_path)

        if self.io_mode == 'lmdb':
            with lmdb.open(path=self.io_path,
                           map_size=self.io_size,
                           lock=False) as env:
                with env.begin() as transaction:
                    return [
                        key.decode()
                        for key in transaction.cursor().iternext(keys=True,
                                                                 values=False)
                    ]

    def eegs(self):
        r'''
        Get all EEG signals in the EEGSignalIO.

        Returns:
            list: The list of EEG signals in the EEGSignalIO.
        '''
        return [self.read_eeg(key) for key in self.keys()]

    def to_dict(self):
        r'''
        Convert EEGSignalIO to an in-memory dictionary, where the index of each sample in the database corresponds to the key, and the EEG signal stored in the database corresponds to the value.

        Returns:
            dict: The dict of samples in the EEGSignalIO.
        '''
        return {key: self.read_eeg(key) for key in self.keys()}

    def read_eeg_in_memory(self, key: str) -> any:
        r'''
        Read all the EEGSignalIO into memory, and index the specified EEG signal in memory with the given :obj:`key`.

        .. warning::
            This method will read all the data in EEGSignalIO into memory, which may cause memory overflow. Thus, it is only recommended for fast reading of small-scale datasets.

        Args:
            key (str): The index of the EEG signal to be queried.
            
        Returns:
            any: The EEG signal sample.
        '''
        if self._in_memory is None:
            self._in_memory = self.to_dict()

        return self._in_memory[key]
