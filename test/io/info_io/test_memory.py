import unittest
from test.io.info_io.test_base import _assert_backend

from torcheeg.io import MemoryInfoIO


class TestMemoryInfoIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, MemoryInfoIO)
