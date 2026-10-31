import gc
import multiprocessing as mp
import os
import pickle
import tempfile
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BaseDataset, MNERawDataset
from torcheeg.io import EEGSignalIO, eeg_signal
from torcheeg.model_selection import Subcategory


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
                self.assertEqual(survivor.read_eeg(
                    'after_delete').tolist(), [9])
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
        np.testing.assert_array_equal(
            subsets[0][0][0], np.arange(12).reshape(3, 4))
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
                actual = np.concatenate([signal.numpy()
                                        for signal, _ in batches])
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
            self.assertEqual(
                result, ('ok', np.arange(12).reshape(3, 4).tolist()))
            reopened = EEGSignalIO(self._path, io_mode='lmdb')
            np.testing.assert_array_equal(reopened.read_eeg('sample'),
                                          np.arange(12).reshape(3, 4))
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            queue.close()
            queue.join_thread()


def _read_in_child(store, connection):
    try:
        inherited = store._io._env_obj
        sample = store.read_eeg('sample')
        connection.send(('ok', sample.tolist(), store._io._env is not inherited,
                         os.getpid()))
    except Exception as error:
        connection.send(('error', repr(error)))
    finally:
        connection.close()


class TestLMDBEnvironment(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)

    def _make_store(self, path=None, io_size=1048576):
        if path is None:
            path = self._directory.name
        store = EEGSignalIO(path, io_mode='lmdb', io_size=io_size)
        return store

    def test_independent_wrappers_share_environment_and_sample_updates(self):
        first = self._make_store()
        first.write_eeg(np.arange(8), key='sample')
        second = self._make_store()
        self.assertIs(first._io._env, second._io._env)
        np.testing.assert_array_equal(second.read_eeg('sample'), np.arange(8))
        second.write_eeg(np.arange(8) + 1, key='sample')
        np.testing.assert_array_equal(
            first.read_eeg('sample'), np.arange(8) + 1)

    def test_canonical_paths_share_one_environment(self):
        root = os.path.join(self._directory.name, 'store')
        alias = os.path.join(self._directory.name, 'alias')
        first = self._make_store(root)
        os.symlink(root, alias, target_is_directory=True)
        second = self._make_store(alias)
        third = self._make_store(os.path.join(root, '.'))
        self.assertIs(first._io._env, second._io._env)
        self.assertIs(first._io._env, third._io._env)
        first.write_eeg(np.arange(4), key='sample')
        np.testing.assert_array_equal(
            second.read_eeg('sample'), third.read_eeg('sample'))

    def test_pickle_round_trip_in_same_process_shares_environment(self):
        first = self._make_store()
        first.write_eeg(np.arange(4), key='sample')
        second = pickle.loads(pickle.dumps(first))
        self.assertIs(first._io._env, second._io._env)
        first.write_eeg(np.arange(4) + 2, key='sample')
        del first
        gc.collect()
        np.testing.assert_array_equal(
            second.read_eeg('sample'), np.arange(4) + 2)

    def test_legacy_backend_pickle_state_can_be_restored(self):
        store = self._make_store()
        store.write_eeg(np.arange(4), key='sample')
        restored = type(store._io).__new__(type(store._io))
        restored.__setstate__(
            {'io_path': self._directory.name, 'io_size': 1048576})
        self.assertIs(restored._env, store._io._env)
        np.testing.assert_array_equal(
            restored.read_eeg('sample'), np.arange(4))

    def test_environment_is_released_after_last_owner(self):
        first = EEGSignalIO(self._directory.name, io_mode='lmdb')
        second = EEGSignalIO(self._directory.name, io_mode='lmdb')
        self.assertIs(first._io._env, second._io._env)
        environment = weakref.ref(first._io._env)
        first.write_eeg(np.arange(4), key='sample')
        del first
        gc.collect()
        self.assertIsNotNone(environment())
        np.testing.assert_array_equal(second.read_eeg('sample'), np.arange(4))
        del second
        gc.collect()
        self.assertIsNone(environment())
        reopened = self._make_store()
        np.testing.assert_array_equal(
            reopened.read_eeg('sample'), np.arange(4))

    def test_repeated_unique_paths_do_not_retain_environments(self):
        references = []
        for index in range(12):
            store = EEGSignalIO(os.path.join(
                self._directory.name, str(index)), io_mode='lmdb')
            store.write_eeg(np.array([index]), key='sample')
            references.append(weakref.ref(store._io._env))
            del store
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_threaded_acquisition_opens_one_environment(self):
        original = self._make_store()
        original.write_eeg(np.arange(16), key='sample')

        def read_store(_):
            store = EEGSignalIO(self._directory.name, io_mode='lmdb')
            for _ in range(8):
                np.testing.assert_array_equal(
                    store.read_eeg('sample'), np.arange(16))
            return store

        with ThreadPoolExecutor(max_workers=8) as executor:
            stores = list(executor.map(read_store, range(24)))
        self.assertTrue(
            all(store._io._env is original._io._env for store in stores))

    def test_map_growth_uses_shared_capacity_for_each_wrapper(self):
        first = self._make_store(io_size=32768)
        second = self._make_store(io_size=32768)
        first_sample = np.arange(250000, dtype=np.float32)
        second_sample = np.arange(500000, dtype=np.float32)
        first.write_eeg(first_sample, key='first')
        capacity = first._io._env.info()['map_size']
        second.write_eeg(second_sample, key='second')
        self.assertGreater(second._io._env.info()['map_size'], capacity)
        np.testing.assert_array_equal(first.read_eeg('second'), second_sample)
        np.testing.assert_array_equal(second.read_eeg('first'), first_sample)

    def test_failed_open_does_not_poison_a_later_valid_store(self):
        path = os.path.join(self._directory.name, 'store')
        with open(path, 'w') as file:
            file.write('not a directory')
        with self.assertRaises(OSError):
            EEGSignalIO(path, io_mode='lmdb')
        os.remove(path)
        store = self._make_store(path)
        store.write_eeg(np.arange(4), key='sample')
        np.testing.assert_array_equal(store.read_eeg('sample'), np.arange(4))

    def _run_child(self, method, stale_capacity=False, hold_cache_lock=False):
        if stale_capacity:
            first = self._make_store(io_size=32768)
            store = self._make_store(io_size=32768)
            sample = np.arange(250000, dtype=np.float32)
            first.write_eeg(sample, key='sample')
        else:
            store = self._make_store()
            sample = np.arange(4)
            store.write_eeg(sample, key='sample')
        parent_environment = store._io._env
        context = mp.get_context(method)
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_read_in_child, args=(store, sender))
        if hold_cache_lock:
            lock = eeg_signal._LMDB_ENV_LOCK
            lock.acquire()
            try:
                process.start()
            finally:
                lock.release()
        else:
            process.start()
        sender.close()
        try:
            self.assertTrue(receiver.poll(
                30), 'Child did not return its sample.')
            result = receiver.recv()
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result[0], 'ok', result)
            self.assertEqual(result[1], sample.tolist())
            self.assertNotEqual(result[3], os.getpid())
            if method == 'fork':
                self.assertTrue(
                    result[2], 'Child reused the inherited LMDB environment.')
            self.assertIs(store._io._env, parent_environment)
            np.testing.assert_array_equal(store.read_eeg('sample'), sample)
            store.write_eeg(np.arange(4) + 1, key='sample')
            np.testing.assert_array_equal(
                store.read_eeg('sample'), np.arange(4) + 1)
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            receiver.close()

    def test_spawned_process_reads_a_fresh_environment(self):
        self._run_child('spawn')

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'requires fork')
    def test_forked_process_reopens_inherited_environment(self):
        self._run_child('fork')

    def test_child_reopens_after_another_wrapper_grows_the_map(self):
        for method in ('spawn', 'fork', 'forkserver'):
            if method in mp.get_all_start_methods():
                with self.subTest(method=method):
                    self._run_child(method, stale_capacity=True)

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'requires fork')
    def test_forked_child_does_not_wait_on_inherited_cache_lock(self):
        self._run_child('fork', hold_cache_lock=True)

    def test_mne_dataset_cache_reuse_and_worker_loading(self):
        methods = [method for method in ('spawn', 'fork', 'forkserver')
                   if method in mp.get_all_start_methods()]
        eeg = np.arange(800, dtype=float).reshape(2, 400)
        raw = mne.io.RawArray(eeg, mne.create_info(
            ['C3', 'C4'], 100, 'eeg'), verbose='ERROR')
        expected = torch.tensor(np.stack([eeg[:, start:start + 100]
                                         for _ in range(2)
                                         for start in range(0, 400, 100)]), dtype=torch.float32)
        for threshold in (0, 128):
            for method in methods:
                with self.subTest(lazy_threshold=threshold, method=method):
                    parameters = dict(
                        raw_list=[raw, raw],
                        info_list=[{'subject_id': '01', 'trial_id': index}
                                   for index in range(2)],
                        chunk_size=100, io_mode='lmdb', lazy_threshold=threshold,
                        io_path=os.path.join(
                            self._directory.name, f'dataset_{threshold}_{method}'),
                        online_transform=transforms.ToTensor(),
                        label_transform=transforms.Select('trial_id'), verbose=False)
                    dataset = MNERawDataset(**parameters)
                    reused = MNERawDataset(**parameters)
                    loader = DataLoader(reused, batch_size=2, num_workers=2,
                                        multiprocessing_context=method, timeout=30)
                    batches = list(loader)
                    torch.testing.assert_close(
                        torch.cat([batch[0] for batch in batches]), expected)
                    torch.testing.assert_close(torch.cat([batch[1] for batch in batches]),
                                               torch.tensor([0] * 4 + [1] * 4))
                    np.testing.assert_array_equal(
                        dataset[2][0].numpy(), eeg[:, 200:300])

    def test_environment_opens_only_on_first_access(self):
        with patch.object(eeg_signal.lmdb, 'open', wraps=eeg_signal.lmdb.open) as open_env:
            store = self._make_store()
            self.assertIsNone(store._io._env_obj)
            self.assertIsNone(store._io._env_pid)
            open_env.assert_not_called()
            self.assertEqual(len(store), 0)
            self.assertEqual(store._io._env_pid, os.getpid())
            open_env.assert_called_once()
            self.assertEqual(len(store), 0)
            open_env.assert_called_once()

    def test_pickle_restoration_defers_environment_acquisition(self):
        store = self._make_store()
        store.write_eeg(np.arange(4), key='sample')
        state = store._io.__getstate__()
        self.assertIsNone(state['_env_obj'])
        self.assertIsNone(state['_env_pid'])
        with patch.object(eeg_signal, '_get_lmdb_environment',
                          wraps=eeg_signal._get_lmdb_environment) as acquire:
            restored = pickle.loads(pickle.dumps(store))
            self.assertIsNone(restored._io._env_obj)
            self.assertIsNone(restored._io._env_pid)
            acquire.assert_not_called()
            np.testing.assert_array_equal(
                restored.read_eeg('sample'), np.arange(4))
            acquire.assert_called_once()
            self.assertIs(restored._io._env_obj, store._io._env_obj)


if __name__ == '__main__':
    unittest.main()
