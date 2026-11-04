from copy import deepcopy

import pandas as pd

from .base import BaseInfoIO


class MemoryInfoIO(BaseInfoIO):
    persistent = False

    def __init__(self, io_path=None):
        self.io_path = None
        self._rows = []

    def write_info(self, obj):
        if not isinstance(obj, dict):
            raise TypeError('obj must be a dictionary.')
        self._rows.append(deepcopy(obj))
        return len(self._rows) - 1

    def read_all(self):
        return pd.DataFrame(deepcopy(self._rows))

    def clone(self, io_path=None):
        return type(self)()
