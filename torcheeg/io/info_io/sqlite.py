import json
import os
import sqlite3
from contextlib import closing

import pandas as pd

from .base import BaseInfoIO, _dump_value


class SQLiteInfoIO(BaseInfoIO):
    """Store metadata rows in SQLite; connections remain local to each call."""

    DEFAULT_FILENAME = 'info.sqlite'

    def __init__(self, io_path=None):
        super().__init__(io_path)
        os.makedirs(os.path.dirname(
            os.path.abspath(self.io_path)), exist_ok=True)
        with closing(sqlite3.connect(self.io_path)) as connection, connection:
            connection.execute(
                'CREATE TABLE IF NOT EXISTS info (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')

    def write_info(self, obj):
        if not isinstance(obj, dict):
            raise TypeError('obj must be a dictionary.')
        payload = _dump_value(obj)
        with closing(sqlite3.connect(self.io_path)) as connection, connection:
            cursor = connection.execute(
                'INSERT INTO info (data) VALUES (?)', (payload,))
            return cursor.lastrowid - 1

    def read_all(self):
        with closing(sqlite3.connect(self.io_path)) as connection, connection:
            rows = connection.execute(
                'SELECT data FROM info ORDER BY id').fetchall()
        return pd.DataFrame([json.loads(row[0]) for row in rows])
