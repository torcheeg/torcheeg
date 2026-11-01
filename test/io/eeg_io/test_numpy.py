import unittest
from test.io.eeg_io.test_base import _assert_backend

from torcheeg.io import NumpyEEGIO


class TestNumpyEEGIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, NumpyEEGIO)

    def test_memory_mapping_and_copy_on_write(self):
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as path:
            io = NumpyEEGIO(path, mmap_mode='c')
            io.write_eeg(np.zeros((2, 4), dtype=np.float32), 'sample')
            sample = io.read_eeg('sample')
            self.assertIsInstance(sample, np.memmap)
            sample[:] = 1
            np.testing.assert_array_equal(
                io.read_eeg('sample'), np.zeros((2, 4)))
            with self.assertRaises(TypeError):
                io.write_eeg(np.array([object()]), 'object')
            with self.assertRaises(ValueError):
                NumpyEEGIO(path, mmap_mode='w+')

    def test_keys_cannot_escape_storage_directory(self):
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as path:
            io = NumpyEEGIO(path)
            for key in ('../sample', 'a/b', '中文'):
                io.write_eeg(np.arange(3), key)
                np.testing.assert_array_equal(io.read_eeg(key), np.arange(3))
            self.assertEqual(set(io.keys()), {'../sample', 'a/b', '中文'})
