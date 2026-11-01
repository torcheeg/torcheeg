import unittest

import numpy as np
import pandas as pd

from torcheeg.datasets import BaseDataset
from torcheeg.io import MemoryEEGIO, MemoryInfoIO


class TestBaseDataset(unittest.TestCase):

    def test_required_class_metadata(self):
        for name in ('channels', 'grid', 'adjacency'):
            with self.subTest(name=name), self.assertRaisesRegex(NotImplementedError, name):
                getattr(BaseDataset, name)

    def test_io_types(self):
        with self.assertRaises(TypeError):
            BaseDataset(object(), MemoryInfoIO())
        with self.assertRaises(TypeError):
            BaseDataset(MemoryEEGIO(), object())

    def test_metadata_selection_used_by_splitters(self):
        dataset = BaseDataset(MemoryEEGIO(), MemoryInfoIO())
        dataset.extend(
            ({'eeg': np.array([i]), 'info': {'label': i}} for i in range(3)))
        dataset.info = dataset.info.iloc[[2, 0]]
        self.assertEqual([sample['info']['label']
                         for sample in dataset], [2, 0])
        self.assertEqual(dataset.info.index.tolist(), [0, 1])
        with self.assertRaises(TypeError):
            dataset.info = []
        self.assertIn('length=2', repr(dataset))
