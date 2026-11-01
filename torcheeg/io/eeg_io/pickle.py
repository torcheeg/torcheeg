import os
import pickle
from typing import Union
from urllib.parse import quote, unquote

import torch

from ...utils import get_random_dir_path
from .base import BaseEEGIO


class PickleEEGIO(BaseEEGIO):

    def __init__(self, io_path: str = None) -> None:
        self.io_path = get_random_dir_path(
            dir_prefix="eeg") if io_path is None else os.fspath(io_path)

        os.makedirs(self.io_path, exist_ok=True)

    def __len__(self):
        return len(os.listdir(self.io_path))

    def write_eeg(self,
                  eeg: Union[any, torch.Tensor],
                  key: str, record=None) -> str:
        r'''
            Write EEG signal to folder.

            Args:
                eeg (any): EEG signal samples to be written into the folder.
                key (str): Sample key. An existing key replaces its sample.

            Returns:
                str: The key of the written EEG signal.
            '''
        key = self.encode_key(key, record)

        if not isinstance(key, str):
            raise TypeError('key must be a string.')

        if eeg is None:
            raise ValueError(f'Save None to the LMDB with the key {key}!')

        with open(os.path.join(self.io_path, ('%2E' * len(key) if key in ('.', '..') else quote(key, safe=''))), 'wb') as f:
            pickle.dump(eeg, f)

        return key

    def read_eeg(self, key: str, record=None) -> any:
        r'''
            Query the corresponding EEG signal in the folder according to the index.

            Args:
                key (str): The index of the EEG signal to be queried.

            Returns:
                any: The EEG signal sample.
            '''
        key = self.encode_key(key, record)
        try:
            with open(os.path.join(self.io_path, ('%2E' * len(key) if key in ('.', '..') else quote(key, safe=''))), 'rb') as f:
                eeg = pickle.load(f)
        except FileNotFoundError as error:
            raise KeyError(key) from error

        return eeg

    def keys(self):
        r'''
            Get all keys in the PickleEEGIO.

            Returns:
                list: The list of keys in the PickleEEGIO.
            '''
        return [unquote(name) for name in os.listdir(self.io_path)]

    def eegs(self):
        r'''
            Get all EEG signals in the PickleEEGIO.

            Returns:
                list: The list of EEG signals in the PickleEEGIO.
            '''
        return [self.read_eeg(key) for key in self.keys()]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def clone(self, io_path):
        return type(self)(io_path=io_path)
