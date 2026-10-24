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
            io.write_eeg(eeg)

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg)

    def test_len(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg)

            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg)

            self.assertEqual(len(io), 2)

    def test_read_eeg(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg)

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_types(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = torch.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg)

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg)

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_shapes(self):
        io_io_path = os.path.join(self._io_path, 'signals')

        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = EEGSignalIO(io_path=os.path.join(io_io_path, io_mode), io_mode=io_mode)

            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg)

            eeg = np.random.randn(128, 9, 9)
            io_eeg_index = io.write_eeg(eeg)

            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))


if __name__ == '__main__':
    unittest.main()