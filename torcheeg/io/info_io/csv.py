import csv
import json
import os
import tempfile

import pandas as pd

from .base import BaseInfoIO, _dump_value


class CSVInfoIO(BaseInfoIO):
    """Store JSON-encoded CSV cells to preserve metadata value types."""

    DEFAULT_FILENAME = 'info.csv'

    def __init__(self, io_path=None):
        super().__init__(io_path)
        os.makedirs(os.path.dirname(
            os.path.abspath(self.io_path)), exist_ok=True)
        with open(self.io_path, 'a', encoding='utf-8'):
            pass
        self._length = None
        self._columns = None

    def write_info(self, obj):
        if not isinstance(obj, dict):
            raise TypeError('obj must be a dictionary.')
        if not obj:
            raise ValueError('obj must contain at least one field.')
        if self._length is None:
            self._length = len(self.read_all())
        empty = os.path.getsize(self.io_path) == 0
        columns = self._columns
        if columns is None:
            if empty:
                columns = []
            else:
                with open(self.io_path, newline='', encoding='utf-8') as file:
                    columns = next(csv.reader(file))
        added = [name for name in obj if name not in columns]
        columns = columns + added
        row = {name: _dump_value(obj.get(name)) for name in columns}
        if added and not empty:
            with open(self.io_path, newline='', encoding='utf-8') as file:
                previous = list(csv.DictReader(file))
            descriptor, temporary = tempfile.mkstemp(
                prefix='.info-', dir=os.path.dirname(os.path.abspath(self.io_path)))
            try:
                with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    for previous_row in previous:
                        previous_row.update({name: 'null' for name in added})
                        writer.writerow(previous_row)
                os.replace(temporary, self.io_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        with open(self.io_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if empty:
                writer.writeheader()
            writer.writerow(row)
        self._columns = columns
        key = self._length
        self._length += 1
        return key

    def read_all(self):
        with open(self.io_path, newline='', encoding='utf-8') as file:
            rows = []
            for row in csv.DictReader(file):
                values = {}
                for name, value in row.items():
                    try:
                        values[name] = json.loads(value)
                    except (ValueError, TypeError):
                        values[name] = value
                rows.append(values)
        return pd.DataFrame(rows)
