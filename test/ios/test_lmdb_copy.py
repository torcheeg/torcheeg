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


def read_in_spawned_process(io, queue):
    try:
        queue.put(('ok', io.read_eeg('sample').tolist()))
    except Exception as error:
        queue.put(('error', repr(error)))


class TestLMDBCopy(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'signals')

    def make_io(self):
        io = EEGSignalIO(self.path, io_mode='lmdb')
        io.write_eeg(np.arange(12).reshape(3, 4), key='sample')
        return io

    def test_copy_does_not_reopen_environment(self):
        io = self.make_io()
        duplicate = copy(io)
        self.assertIsNot(io, duplicate)
        self.assertIs(io._io, duplicate._io)
        np.testing.assert_array_equal(duplicate.read_eeg('sample'), np.arange(12).reshape(3, 4))

    def test_either_wrapper_can_be_deleted_first(self):
        for delete_original in (False, True):
            with self.subTest(delete_original=delete_original):
                io = self.make_io()
                duplicate = copy(io)
                if delete_original:
                    del io
                    survivor = duplicate
                else:
                    del duplicate
                    survivor = io
                gc.collect()
                np.testing.assert_array_equal(survivor.read_eeg('sample'), np.arange(12).reshape(3, 4))
                survivor.write_eeg(np.array([9]), key='after_delete')
                self.assertEqual(survivor.read_eeg('after_delete').tolist(), [9])
                del survivor
                if delete_original:
                    del duplicate
                else:
                    del io
                gc.collect()

    def test_writes_and_map_growth_are_visible_through_both_wrappers(self):
        io = self.make_io()
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
        io = self.make_io()
        copies = [copy(io) for _ in range(5)]
        del io
        survivor = copies.pop()
        copies.clear()
        gc.collect()
        np.testing.assert_array_equal(survivor.read_eeg('sample'), np.arange(12).reshape(3, 4))
        del survivor
        gc.collect()
        reopened = EEGSignalIO(self.path, io_mode='lmdb')
        self.assertEqual(len(reopened), 1)

    def test_eager_dataset_copy_preserves_signal_and_independent_metadata(self):
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.eeg_io_router = {'_record_0': self.make_io()}
        dataset.eeg_signal_paths = {}
        dataset.info = pd.DataFrame([{'_record_id': '_record_0', 'clip_id': 'sample', 'subject_id': 1}])
        duplicate = copy(dataset)
        duplicate.info.loc[0, 'subject_id'] = 2
        self.assertEqual(dataset.info.loc[0, 'subject_id'], 1)
        del dataset
        gc.collect()
        signal, info = duplicate[0]
        np.testing.assert_array_equal(signal, np.arange(12).reshape(3, 4))
        self.assertEqual(info['subject_id'], 2)

    def test_conversion_of_one_wrapper_does_not_close_the_other(self):
        io = self.make_io()
        duplicate = copy(io)
        duplicate.to_memory()
        self.assertEqual(duplicate.io_mode, 'memory')
        self.assertEqual(io.io_mode, 'lmdb')
        del duplicate
        gc.collect()
        np.testing.assert_array_equal(io.read_eeg('sample'), np.arange(12).reshape(3, 4))

    def test_pickled_copy_can_be_read_in_spawned_process(self):
        io = self.make_io()
        duplicate = copy(io)
        context = mp.get_context('spawn')
        queue = context.Queue()
        worker = context.Process(target=read_in_spawned_process, args=(duplicate, queue))
        worker.start()
        try:
            result = queue.get(timeout=60)
            worker.join(timeout=10)
            self.assertEqual(worker.exitcode, 0)
            self.assertEqual(result, ('ok', np.arange(12).reshape(3, 4).tolist()))
            np.testing.assert_array_equal(io.read_eeg('sample'), np.arange(12).reshape(3, 4))
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            queue.close()


if __name__ == '__main__':
    unittest.main()
