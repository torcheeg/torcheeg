import unittest
from test.io.eeg_io.test_base import _assert_backend

from torcheeg.io import MemoryEEGIO


class TestMemoryEEGIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, MemoryEEGIO)
