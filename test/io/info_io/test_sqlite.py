import unittest
from test.io.info_io.test_base import _assert_backend

from torcheeg.io import SQLiteInfoIO


class TestSQLiteInfoIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, SQLiteInfoIO)
