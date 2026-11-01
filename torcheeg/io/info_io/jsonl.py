import json
import os

import pandas as pd

from .base import BaseInfoIO, _dump_value


class JSONLInfoIO(BaseInfoIO):
    """Store one metadata dictionary per line."""

    DEFAULT_FILENAME = 'info.jsonl'

    def __init__(self, io_path=None):
        super().__init__(io_path)
        os.makedirs(os.path.dirname(
            os.path.abspath(self.io_path)), exist_ok=True)
        with open(self.io_path, 'a', encoding='utf-8'):
            pass
        self._length = None

    def write_info(self, obj):
        if not isinstance(obj, dict):
            raise TypeError('obj must be a dictionary.')
        payload = _dump_value(obj)
        if self._length is None:
            self._length = len(self.read_all())
        with open(self.io_path, 'a', encoding='utf-8') as file:
            file.write(payload + '\n')
        key = self._length
        self._length += 1
        return key

    def read_all(self):
        with open(self.io_path, encoding='utf-8') as file:
            return pd.DataFrame([json.loads(line) for line in file if line.strip()])
