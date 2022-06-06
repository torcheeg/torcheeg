import os
import random
import unittest
import shutil

import numpy as np

from torcheeg.io import EEGSignalIO


class TestEEGSignalIO(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./outputs/')
        os.mkdir('./outputs/')

    def test_init(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = EEGSignalIO(cache_path=io_cache_path)
        self.assertEqual(len(io), 0)

        self.assertTrue(os.path.exists(io_cache_path))
        io = EEGSignalIO(cache_path=io_cache_path)
        self.assertEqual(len(io), 0)

    def test_write_eeg(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = EEGSignalIO(cache_path=io_cache_path)

        eeg = np.random.randn(32, 128)
        io.write_eeg(eeg)

        eeg = np.random.randn(32, 128)
        io.write_eeg(eeg)

    def test_len(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = EEGSignalIO(cache_path=io_cache_path)

        eeg = np.random.randn(32, 128)
        io.write_eeg(eeg)

        eeg = np.random.randn(32, 128)
        io.write_eeg(eeg)

        self.assertEqual(len(io), 2)

    def test_read_eeg(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = EEGSignalIO(cache_path=io_cache_path)

        eeg = np.random.randn(32, 128)
        io_eeg_idx = io.write_eeg(eeg)

        io_eeg = io.read_eeg(io_eeg_idx)
        self.assertTrue(np.array_equal(eeg, io_eeg))


if __name__ == '__main__':
    unittest.main()