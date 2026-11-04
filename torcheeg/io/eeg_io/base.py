import json
from copy import copy


class BaseEEGIO:
    """Store signals by string key, optionally scoped to a record."""

    persistent = True

    @staticmethod
    def encode_key(key, record=None):
        if not isinstance(key, str):
            raise TypeError('key must be a string.')
        if not key:
            raise ValueError('key must not be empty.')
        return key if record is None else json.dumps([record, key], separators=(',', ':'))

    def read_eeg(self, key, record=None):
        raise NotImplementedError(
            "Method read_eeg is not implemented in class " + self.__class__.__name__)

    def write_eeg(self, eeg, key, record=None):
        raise NotImplementedError(
            "Method write_eeg is not implemented in class " + self.__class__.__name__)

    def keys(self):
        raise NotImplementedError(
            "Method keys is not implemented in class " + self.__class__.__name__)

    def __len__(self):
        return len(self.keys())

    def eegs(self):
        return [self.read_eeg(key) for key in self.keys()]

    def close(self):
        """Release this wrapper's resources."""

    def clone(self, io_path):
        result = copy(self)
        result.io_path = io_path
        return result
