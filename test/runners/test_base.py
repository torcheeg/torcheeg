import unittest

import numpy as np

from torcheeg.datasets import BaseData, MemoryDataset
from torcheeg.runners import BaseRunner, SequentialRunner


class _WindowData(BaseData):
    def set_records(self, **kwargs):
        return [0, 1, 2]

    def read_record(self, record, **kwargs):
        return {}

    def process_record(self, record, **kwargs):
        buffer = np.zeros((2, 4))
        for index in range(8):
            buffer[:] = index
            yield {'eeg': buffer, 'info': {'subject_id': f'{record:03}', 'index': index}}


class TestBaseRunner(unittest.TestCase):
    def test_required_iterate_and_inputs(self):
        with self.assertRaises(NotImplementedError):
            BaseRunner()(_WindowData())
        with self.assertRaises(TypeError):
            SequentialRunner()(object())
        for value in (True, 1.5):
            with self.assertRaises(TypeError):
                BaseRunner(value)
        with self.assertRaises(ValueError):
            BaseRunner(-1)

    def test_independent_snapshots_and_streaming_storage(self):
        dataset = MemoryDataset()
        dataset.extend(SequentialRunner()(_WindowData()))
        self.assertEqual(len(dataset), 24)
        np.testing.assert_array_equal(dataset[0]['eeg'], np.zeros((2, 4)))
        np.testing.assert_array_equal(dataset[7]['eeg'], np.full((2, 4), 7))
