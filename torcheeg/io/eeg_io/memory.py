from typing import Union

import torch

from .base import BaseEEGIO


class MemoryEEGIO(BaseEEGIO):

    persistent = False

    def __init__(self, io_path=None):
        self.io_path = None
        self._memory = {}

    def __len__(self):
        return len(self._memory)

    def keys(self):
        r'''
        Get all keys in the MemoryEEGIO.

        Returns:
            list: The list of keys in the MemoryEEGIO.
        '''
        return list(self._memory.keys())

    def eegs(self):
        return list(self._memory.values())

    def read_eeg(self, key: str, record=None) -> any:
        r'''
        Read all the MemoryEEGIO into memory, and index the specified EEG signal in memory with the given :obj:`key`.

        Args:
            key (str): The index of the EEG signal to be queried.

        Returns:
            any: The EEG signal sample.
        '''
        key = self.encode_key(key, record)
        if key not in self._memory:
            raise KeyError(
                f'Unable to index the EEG signal sample with key {key}!')

        return self._memory[key]

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str, record=None) -> str:
        r'''
        Write EEG signal to memory.

        Args:
            eeg (any): EEG signal samples to be written into the database.
            key (str): Sample key. An existing key replaces its sample.
        '''
        key = self.encode_key(key, record)
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

    def clone(self, io_path=None):
        return type(self)()
