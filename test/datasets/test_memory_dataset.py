import unittest

import numpy as np

from torcheeg.datasets import MemoryDataset
from torcheeg.io import MemoryEEGIO, MemoryInfoIO


class TestMemoryDataset(unittest.TestCase):

    def test_default_storage_is_independent(self):
        left, right = (MemoryDataset(), MemoryDataset())
        left.append({'eeg': np.ones((2, 4)), 'info': {'subject_id': '001'}})
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 0)
        self.assertEqual(left[0]['info']['subject_id'], '001')

    def test_explicit_storage(self):
        eeg, info = (MemoryEEGIO(), MemoryInfoIO())
        dataset = MemoryDataset(eeg, info)
        dataset.append({'eeg': np.zeros(4), 'info': {}})
        self.assertEqual(len(MemoryDataset(eeg, info)), 1)
        with self.assertRaises(TypeError):
            MemoryDataset(eeg_io=object())
