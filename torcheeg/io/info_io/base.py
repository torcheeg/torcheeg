import json
import os

import numpy as np
import pandas as pd

from ...utils import get_random_dir_path


def _encode_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    raise TypeError(f'Unsupported metadata type {type(value).__name__}.')


def _dump_value(value):
    return json.dumps(value, default=_encode_value, ensure_ascii=False)


class BaseInfoIO:
    """Store sample metadata in insertion order."""

    persistent = True
    DEFAULT_FILENAME = 'info'

    def __init__(self, io_path=None):
        self.io_path = os.path.join(get_random_dir_path(
            dir_prefix='info'), self.DEFAULT_FILENAME) if io_path is None else os.fspath(io_path)

    def write_info(self, obj):
        raise NotImplementedError(
            "Method write_info is not implemented in class " + self.__class__.__name__)

    def read_all(self):
        raise NotImplementedError(
            "Method read_all is not implemented in class " + self.__class__.__name__)

    def read_info(self, key):
        return self.read_all().iloc[key]

    def __getitem__(self, index):
        info = self.read_all()
        if isinstance(index, str):
            return info[index]
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
            return info.iloc[int(index)].to_dict()
        if isinstance(index, slice):
            return info.iloc[index].reset_index(drop=True)
        positions = np.asarray(index)
        if positions.ndim != 1:
            raise ValueError('Metadata indices must be one-dimensional.')
        if positions.dtype.kind == 'b':
            if len(positions) != len(info):
                raise ValueError(
                    'Boolean mask length must equal metadata length.')
        elif positions.size and positions.dtype.kind not in 'iu':
            raise TypeError('Metadata indices must be integers or booleans.')
        elif not positions.size:
            positions = positions.astype(np.intp)
        return info.iloc[positions].reset_index(drop=True)

    def __len__(self):
        return len(self.read_all())

    def close(self):
        """Release this wrapper's resources."""

    def clone(self, io_path):
        return type(self)(io_path)
