import gc
import multiprocessing as mp
import os
import tempfile
import unittest
from copy import copy

import numpy as np
import pandas as pd

from torcheeg.datasets import BaseDataset
from torcheeg.io import EEGSignalIO
from torcheeg.model_selection import Subcategory
from torch.utils.data import DataLoader


def _read_in_spawned_process(io, queue, release):
    try:
        if not release.wait(timeout=30):
            raise RuntimeError('Parent did not release the worker.')
        queue.put(('ok', io.read_eeg('sample').tolist()))
    except Exception as error:
        queue.put(('error', repr(error)))


class TestLMDBCopy(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._path = os.path.join(self._directory.name, 'signals')

    def _make_io(self):
        io = EEGSignalIO(self._path, io_mode='lmdb')
        io.write_eeg(np.arange(12).reshape(3, 4), key='sample')
        return io

    def test_copy_does_not_reopen_environment(self):
        io = self._make_io()
        duplicate = copy(io)
        self.assertIsNot(io, duplicate)
        self.assertIs(io._io, duplicate._io)
        np.testing.assert_array_equal(duplicate.read_eeg('sample'),
                                      np.arange(12).reshape(3, 4))

    def test_either_wrapper_can_be_deleted_first(self):
        for delete_original in (False, True):
            with self.subTest(delete_original=delete_original):
                io = self._make_io()
                duplicate = copy(io)
                if delete_original:
                    del io
                    survivor = duplicate
                else:
                    del duplicate
                    survivor = io
                gc.collect()
                np.testing.assert_array_equal(survivor.read_eeg('sample'),
                                              np.arange(12).reshape(3, 4))
                survivor.write_eeg(np.array([9]), key='after_delete')
                self.assertEqual(survivor.read_eeg('after_delete').tolist(), [9])
                del survivor
                if delete_original:
                    del duplicate
                else:
                    del io
                gc.collect()

    def test_writes_and_map_growth_are_visible_through_both_wrappers(self):
        io = self._make_io()
        duplicate = copy(io)
        signal = np.arange(400000, dtype=np.float64)
        duplicate.write_eeg(signal, key='large')
        self.assertGreater(duplicate._io.io_size, 1048576)
        np.testing.assert_array_equal(io.read_eeg('large'), signal)
        io.write_eeg(np.array([7]), key='original')
        self.assertEqual(duplicate.read_eeg('original').tolist(), [7])
        self.assertEqual(set(io.keys()), {'sample', 'large', 'original'})
        self.assertEqual(len(duplicate), 3)

    def test_repeated_copies_keep_backend_alive_until_last_reference(self):
        io = self._make_io()
        copies = [copy(io) for _ in range(5)]
        del io
        survivor = copies.pop()
        copies.clear()
        gc.collect()
        np.testing.assert_array_equal(survivor.read_eeg('sample'),
                                      np.arange(12).reshape(3, 4))
        del survivor
        gc.collect()
        reopened = EEGSignalIO(self._path, io_mode='lmdb')
        self.assertEqual(len(reopened), 1)

    def test_eager_dataset_copy_preserves_signal_and_independent_metadata(self):
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.eeg_io_router = {'_record_0': self._make_io()}
        dataset.eeg_signal_paths = {}
        dataset.info = pd.DataFrame([{
            '_record_id': '_record_0',
            'clip_id': 'sample',
            'subject_id': 1
        }])
        duplicate = copy(dataset)
        duplicate.info.loc[0, 'subject_id'] = 2
        self.assertEqual(dataset.info.loc[0, 'subject_id'], 1)
        del dataset
        gc.collect()
        signal, info = duplicate[0]
        np.testing.assert_array_equal(signal, np.arange(12).reshape(3, 4))
        self.assertEqual(info['subject_id'], 2)

    def test_eager_subcategory_split_shares_signals(self):
        io = self._make_io()
        io.write_eeg(np.array([42]), key='other')
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.eeg_io_router = {'_record_0': io}
        dataset.eeg_signal_paths = {}
        dataset.info = pd.DataFrame({
            '_record_id': ['_record_0', '_record_0'],
            'clip_id': ['sample', 'other'],
            'category': [0, 1]
        })
        subsets = list(
            Subcategory(criteria='category',
                        split_path=os.path.join(self._directory.name,
                                                'split')).split(dataset))
        del dataset, io
        gc.collect()
        np.testing.assert_array_equal(subsets[0][0][0], np.arange(12).reshape(3, 4))
        np.testing.assert_array_equal(subsets[1][0][0], [42])

    def test_copied_dataset_can_be_read_by_multiple_workers(self):
        io = self._make_io()
        io.write_eeg(np.arange(12).reshape(3, 4) + 100, key='other')
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.eeg_io_router = {'_record_0': io}
        dataset.eeg_signal_paths = {}
        dataset.info = pd.DataFrame({
            '_record_id': ['_record_0', '_record_0'],
            'clip_id': ['sample', 'other'],
            'subject_id': [1, 2]
        })
        duplicate = copy(dataset)
        del dataset, io
        gc.collect()
        expected = np.stack(
            (np.arange(12).reshape(3, 4), np.arange(12).reshape(3, 4) + 100))
        for method in mp.get_all_start_methods():
            with self.subTest(start_method=method):
                loader = DataLoader(duplicate,
                                    batch_size=1,
                                    num_workers=2,
                                    multiprocessing_context=method)
                batches = list(loader)
                actual = np.concatenate([signal.numpy() for signal, _ in batches])
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(duplicate[0][0], expected[0])

    def test_conversion_of_one_wrapper_does_not_close_the_other(self):
        io = self._make_io()
        duplicate = copy(io)
        duplicate.to_memory()
        self.assertEqual(duplicate.io_mode, 'memory')
        self.assertEqual(io.io_mode, 'lmdb')
        del duplicate
        gc.collect()
        np.testing.assert_array_equal(io.read_eeg('sample'),
                                      np.arange(12).reshape(3, 4))

    def test_spawned_reader_survives_parent_release(self):
        io = self._make_io()
        duplicate = copy(io)
        context = mp.get_context('spawn')
        queue = context.Queue()
        release = context.Event()
        worker = context.Process(target=_read_in_spawned_process,
                                 args=(duplicate, queue, release))
        worker.start()
        del io, duplicate
        gc.collect()
        release.set()
        try:
            result = queue.get(timeout=60)
            worker.join(timeout=10)
            self.assertEqual(worker.exitcode, 0)
            self.assertEqual(result, ('ok', np.arange(12).reshape(3, 4).tolist()))
            reopened = EEGSignalIO(self._path, io_mode='lmdb')
            np.testing.assert_array_equal(reopened.read_eeg('sample'),
                                          np.arange(12).reshape(3, 4))
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            queue.close()
            queue.join_thread()


if __name__ == '__main__':
    unittest.main()
