import unittest
from test.io.eeg_io.test_base import _assert_backend

from torcheeg.io import ZarrEEGIO


class TestZarrEEGIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, ZarrEEGIO)
