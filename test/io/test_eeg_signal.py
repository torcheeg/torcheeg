import os
import tempfile
import unittest

import numpy as np
import torch

from torcheeg.io import EEGSignalIO


class TestEEGSignalIO(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._io_path = self._directory.name

    def test_init(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)
            self.assertEqual(len(io), 0)

            self.assertTrue(os.path.exists(io_io_path))
            del io
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)
            self.assertEqual(len(io), 0)

    def test_write_eeg(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='0')

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='1')

    def test_len(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='0')

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='1')

            self.assertEqual(len(io), 2)

    def test_read_eeg(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_types(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = torch.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='1')

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_shapes(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')

            eeg = np.random.randn(128, 9, 9)
            io_eeg_index = io.write_eeg(eeg, key='1')

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))


    def test_write_eeg_requires_string_key(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = EEGSignalIO(path, io_mode=mode)
                sample = np.array([10])
                with self.assertRaises(TypeError):
                    io.write_eeg(sample)
                for key in (None, 0):
                    with self.assertRaisesRegex(TypeError, 'key must be a string'):
                        io.write_eeg(sample, key=key)
                    with self.assertRaisesRegex(TypeError, 'key must be a string'):
                        io._io.write_eeg(sample, key=key)
                self.assertEqual(len(io), 0)
                del io

    def test_write_eeg_preserves_other_keys_and_allows_replacement(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = EEGSignalIO(path, io_mode=mode)
                io.write_eeg(np.array([10]), key='0')
                io.write_eeg(np.array([20]), key='2')
                self.assertEqual(io.write_eeg(np.array([30]), key='new'), 'new')
                io.write_eeg(np.array([22]), key='2')
                self.assertEqual(len(io), 3)
                for key, value in (('0', 10), ('2', 22), ('new', 30)):
                    np.testing.assert_array_equal(io.read_eeg(key), [value])
                del io

    def test_lmdb_map_growth_preserves_explicit_key(self):
        with tempfile.TemporaryDirectory() as path:
            io = EEGSignalIO(path, io_mode='lmdb', io_size=32768)
            sample = np.arange(250000, dtype=np.float32)
            self.assertEqual(io.write_eeg(sample, key='large'), 'large')
            np.testing.assert_array_equal(io.read_eeg('large'), sample)
            del io


if __name__ == '__main__':
    unittest.main()