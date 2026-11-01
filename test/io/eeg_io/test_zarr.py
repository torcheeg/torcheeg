import importlib.util
import unittest
from test.io.eeg_io.test_base import _assert_backend

from torcheeg.io import ZarrEEGIO


class TestZarrEEGIO(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('zarr'), 'optional zarr is not installed')
    def test_backend_contract(self):
        _assert_backend(self, ZarrEEGIO)
