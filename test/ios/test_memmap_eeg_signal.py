import concurrent.futures
import multiprocessing as mp
import os
import pickle
import tempfile
import threading
import unittest
from copy import copy
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torcheeg.datasets import MNERawDataset
from torcheeg.io import EEGSignalIO, MemmapEEGSignalIO


def write_samples(path, value, queue):
    try:
        io = MemmapEEGSignalIO(path)
        keys = [io.write_eeg(np.full((3, 4), value * 10 + index)) for index in range(10)]
        queue.put(('ok', keys))
    except Exception as error:
        queue.put(('error', repr(error)))


class StoredSamples(Dataset):
    def __init__(self, io):
        self.io = io
        self.keys = io.keys()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # Returned maps are read-only; make the tensor's array writable.
        return np.array(self.io.read_eeg(self.keys[index]), copy=True)


class TestMemmapEEGSignalIO(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, 'signals')
        self.io = MemmapEEGSignalIO(self.path)

    def test_arrays_preserve_shape_dtype_and_values(self):
        arrays = [np.arange(12, dtype=dtype).reshape(3, 4)
                  for dtype in (np.int16, np.int64, np.float32, np.float64, np.complex64)]
        arrays += [np.array(3.), np.empty((0, 2)), np.ones((2, 3, 4), dtype=bool)]
        for index, array in enumerate(arrays):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                key = self.io.write_eeg(array)
                self.assertEqual(key, str(index))
                actual = self.io.read_eeg(key)
                self.assertIsInstance(actual, np.memmap)
                self.assertEqual(actual.dtype, array.dtype)
                self.assertEqual(actual.shape, array.shape)
                self.assertFalse(actual.flags.writeable)
                np.testing.assert_array_equal(actual, array)

    def test_tensor_requires_grad_is_detached_and_array_like_is_supported(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
        key = self.io.write_eeg(tensor)
        np.testing.assert_array_equal(self.io.read_eeg(key), tensor.detach().numpy())
        self.io.write_eeg([[1, 2], [3, 4]], key='list')
        np.testing.assert_array_equal(self.io.read_eeg('list'), [[1, 2], [3, 4]])

    def test_auto_keys_do_not_overwrite_sparse_or_custom_keys(self):
        self.io.write_eeg([1], key='1')
        self.io.write_eeg([3], key='custom')
        self.assertEqual(self.io.write_eeg([2]), '2')
        np.testing.assert_array_equal(self.io.read_eeg('1'), [1])
        self.assertEqual(self.io.write_pointer, 3)
        self.assertEqual(len(self.io), 3)

    def test_special_keys_roundtrip_without_path_escape_or_encoding_collisions(self):
        keys = ['../outside', '/absolute', 'a/b', 'a%2Fb', '空 格', '..', 'key.npy']
        for value, key in enumerate(keys):
            self.io.write_eeg([value], key=key)
        self.assertEqual(set(self.io.keys()), set(keys))
        self.assertEqual(len(self.io), len(keys))
        for value, key in enumerate(keys):
            self.assertEqual(self.io.read_eeg(key).tolist(), [value])
        self.assertEqual({p.name for p in Path(self.directory.name).iterdir()}, {'signals'})
        self.assertTrue(all(p.is_file() for p in Path(self.path).iterdir()))

    def test_invalid_arrays_keys_and_missing_samples(self):
        with self.assertRaises(RuntimeError):
            self.io.write_eeg(None)
        for array in (np.array([object()], dtype=object), {'label': 1}):
            with self.assertRaises(ValueError):
                self.io.write_eeg(array)
        for key in ('', 2, b'bytes'):
            with self.assertRaises(ValueError):
                self.io.write_eeg([1], key=key)
        with self.assertRaises(RuntimeError):
            self.io.read_eeg('missing')
        self.assertEqual(self.io.keys(), [])

    def test_unrelated_files_and_directories_are_not_samples(self):
        Path(self.path, 'note.txt').touch()
        Path(self.path, 'directory.npy').mkdir()
        self.assertEqual(len(self.io), 0)
        self.io.write_eeg([1], key='a')
        self.assertEqual(self.io.keys(), ['a'])
        self.assertEqual([array.tolist() for array in self.io.eegs()], [[1]])

    def test_multiple_openers_copy_and_pickle_share_persisted_samples(self):
        self.io.write_eeg([1, 2], key='sample')
        for duplicate in (MemmapEEGSignalIO(self.path), copy(self.io), pickle.loads(pickle.dumps(self.io))):
            np.testing.assert_array_equal(duplicate.read_eeg('sample'), [1, 2])
            duplicate.write_eeg([3], key='new')
        np.testing.assert_array_equal(self.io.read_eeg('new'), [3])

    def test_incomplete_write_does_not_publish_a_partial_or_replace_old_sample(self):
        self.io.write_eeg([1], key='sample')
        written, release = threading.Event(), threading.Event()
        original_save = np.save
        def paused_save(file, array, **kwargs):
            original_save(file, array, **kwargs)
            written.set()
            if not release.wait(timeout=20):
                raise RuntimeError('Writer pause timed out')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            with patch('torcheeg.io.memmap_eeg_signal.np.save', side_effect=paused_save):
                future = executor.submit(self.io.write_eeg, [2], 'sample')
                try:
                    self.assertTrue(written.wait(timeout=10))
                    self.assertEqual(self.io.keys(), ['sample'])
                    self.assertEqual(self.io.read_eeg('sample').tolist(), [1])
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=10), 'sample')
        self.assertEqual(self.io.read_eeg('sample').tolist(), [2])

    def test_failed_writer_preserves_previous_sample_cleans_temporary_files_and_retries(self):
        self.io.write_eeg([1], key='sample')
        with patch('torcheeg.io.memmap_eeg_signal.np.save', side_effect=RuntimeError('intentional')):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                self.io.write_eeg([2], key='sample')
        self.assertEqual([p.name for p in Path(self.path).iterdir()], ['sample.npy'])
        self.assertEqual(self.io.read_eeg('sample').tolist(), [1])
        self.io.write_eeg([2], key='sample')
        self.assertEqual(self.io.read_eeg('sample').tolist(), [2])

    def test_existing_writer_lock_times_out_without_deleting_it(self):
        lock = Path(self.path, '.torcheeg-write.lock')
        lock.write_text('active writer')
        with patch('torcheeg.io.memmap_eeg_signal._WRITE_TIMEOUT', 0):
            with self.assertRaisesRegex(RuntimeError, 'Timeout'):
                self.io.write_eeg([1])
        self.assertEqual(lock.read_text(), 'active writer')

    @unittest.skipIf(os.name == 'nt', 'Windows can restrict replacement of mapped files')
    def test_existing_map_keeps_its_data_when_the_key_is_replaced(self):
        self.io.write_eeg([1], key='sample')
        previous = self.io.read_eeg('sample')
        self.io.write_eeg([2, 3], key='sample')
        np.testing.assert_array_equal(previous, [1])
        np.testing.assert_array_equal(self.io.read_eeg('sample'), [2, 3])

    def test_independent_process_writers_allocate_unique_auto_keys(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        workers = [context.Process(target=write_samples, args=(self.path, value, queue)) for value in range(2)]
        for worker in workers:
            worker.start()
        try:
            results = [queue.get(timeout=90) for _ in workers]
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode, 0)
            self.assertTrue(all(result[0] == 'ok' for result in results), results)
            self.assertEqual(len(set(results[0][1] + results[1][1])), 20)
            self.assertEqual(len(self.io), 20)
            self.assertEqual({int(self.io.read_eeg(key)[0, 0]) for key in self.io.keys()}, set(range(20)))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()

    def test_dataloader_workers_read_samples_with_spawn_and_fork(self):
        for index in range(6):
            self.io.write_eeg(np.full((2, 3), index), key=str(index))
        for method in ('spawn', 'fork'):
            if method not in mp.get_all_start_methods():
                continue
            with self.subTest(method=method):
                loader = DataLoader(StoredSamples(self.io), batch_size=2, num_workers=2,
                                    multiprocessing_context=method)
                actual = torch.cat(list(loader)).numpy()
                np.testing.assert_array_equal(actual[:, 0, 0], np.arange(6))

    def test_public_wrapper_conversions_preserve_values_and_keys(self):
        io = EEGSignalIO('unused', io_mode='memory')
        io.write_eeg(np.arange(12).reshape(3, 4), key='sample')
        io.to_memmap(self.path)
        self.assertEqual(io.io_mode, 'memmap')
        self.assertIsInstance(io.read_eeg('sample'), np.memmap)
        duplicate = copy(io)
        del io
        np.testing.assert_array_equal(duplicate.read_eeg('sample'), np.arange(12).reshape(3, 4))
        duplicate.to_memory()
        self.assertEqual(duplicate.io_mode, 'memory')
        np.testing.assert_array_equal(duplicate.read_eeg('sample'), np.arange(12).reshape(3, 4))
        direct = EEGSignalIO(self.path, io_mode='memmap')
        np.testing.assert_array_equal(direct.read_eeg('sample'), np.arange(12).reshape(3, 4))

    def test_mne_dataset_caches_and_reopens_memmap_signals(self):
        eeg = np.arange(2 * 400, dtype=float).reshape(2, 400)
        raw = mne.io.RawArray(eeg, mne.create_info(['C3', 'C4'], 100, 'eeg'), verbose='ERROR')
        parameters = dict(raw_list=[raw], info_list=[{'subject_id': '01', 'trial_id': 0}],
                          chunk_size=100, io_mode='memmap',
                          io_path=os.path.join(self.directory.name, 'dataset'), verbose=False)
        dataset = MNERawDataset(**parameters)
        self.assertEqual(len(dataset), 4)
        signal, info = dataset[2]
        self.assertIsInstance(signal, np.memmap)
        np.testing.assert_array_equal(signal, eeg[:, 200:300])
        duplicate = copy(dataset)
        del dataset
        np.testing.assert_array_equal(duplicate[2][0], signal)
        reused = MNERawDataset(**parameters)
        np.testing.assert_array_equal(reused[2][0], signal)
        self.assertEqual(reused[2][1]['trial_id'], 0)

    def test_failed_conversion_does_not_change_original_backend(self):
        io = EEGSignalIO('unused', io_mode='memory')
        io.write_eeg({'object': 1}, key='object')
        with self.assertRaises(ValueError):
            io.to_memmap(self.path)
        self.assertEqual(io.io_mode, 'memory')
        self.assertEqual(io.read_eeg('object'), {'object': 1})


if __name__ == '__main__':
    unittest.main()
