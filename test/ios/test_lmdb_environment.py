import gc
import multiprocessing
import os
import pickle
import tempfile
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MNERawDataset
from torcheeg.io import EEGSignalIO
from torcheeg.io import eeg_signal


def _read_in_child(store, connection):
    try:
        inherited = store._io.__dict__.get('_environment',
                                           store._io.__dict__.get('_env'))
        sample = store.read_eeg('sample')
        connection.send(('ok', sample.tolist(), store._io._env is not inherited,
                         os.getpid()))
    except Exception as error:
        connection.send(('error', repr(error)))
    finally:
        connection.close()


class TestLMDBEnvironment(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def store(self, path=None, io_size=1048576):
        if path is None:
            path = self.directory.name
        store = EEGSignalIO(path, io_mode='lmdb', io_size=io_size)
        self.addCleanup(store._io._env.close)
        return store

    def test_independent_wrappers_share_environment_and_sample_updates(self):
        first = self.store()
        first.write_eeg(np.arange(8), key='sample')
        second = self.store()
        self.assertIs(first._io._env, second._io._env)
        np.testing.assert_array_equal(second.read_eeg('sample'), np.arange(8))
        second.write_eeg(np.arange(8) + 1, key='sample')
        np.testing.assert_array_equal(first.read_eeg('sample'), np.arange(8) + 1)

    def test_canonical_paths_share_one_environment(self):
        root = os.path.join(self.directory.name, 'store')
        alias = os.path.join(self.directory.name, 'alias')
        first = self.store(root)
        os.symlink(root, alias, target_is_directory=True)
        second = self.store(alias)
        third = self.store(os.path.join(root, '.'))
        self.assertIs(first._io._env, second._io._env)
        self.assertIs(first._io._env, third._io._env)
        first.write_eeg(np.arange(4), key='sample')
        np.testing.assert_array_equal(second.read_eeg('sample'), third.read_eeg('sample'))

    def test_pickle_round_trip_in_same_process_shares_environment(self):
        first = self.store()
        first.write_eeg(np.arange(4), key='sample')
        second = pickle.loads(pickle.dumps(first))
        self.assertIs(first._io._env, second._io._env)
        first.write_eeg(np.arange(4) + 2, key='sample')
        del first
        gc.collect()
        np.testing.assert_array_equal(second.read_eeg('sample'), np.arange(4) + 2)

    def test_legacy_backend_pickle_state_can_be_restored(self):
        store = self.store()
        store.write_eeg(np.arange(4), key='sample')
        restored = type(store._io).__new__(type(store._io))
        restored.__setstate__({'io_path': self.directory.name, 'io_size': 1048576})
        self.assertIs(restored._env, store._io._env)
        np.testing.assert_array_equal(restored.read_eeg('sample'), np.arange(4))

    def test_environment_is_released_after_last_owner(self):
        first = EEGSignalIO(self.directory.name, io_mode='lmdb')
        second = EEGSignalIO(self.directory.name, io_mode='lmdb')
        environment = weakref.ref(first._io._env)
        first.write_eeg(np.arange(4), key='sample')
        del first
        gc.collect()
        self.assertIsNotNone(environment())
        np.testing.assert_array_equal(second.read_eeg('sample'), np.arange(4))
        del second
        gc.collect()
        self.assertIsNone(environment())
        reopened = self.store()
        np.testing.assert_array_equal(reopened.read_eeg('sample'), np.arange(4))

    def test_repeated_unique_paths_do_not_retain_environments(self):
        references = []
        for index in range(12):
            store = EEGSignalIO(os.path.join(self.directory.name, str(index)), io_mode='lmdb')
            store.write_eeg(np.array([index]), key='sample')
            references.append(weakref.ref(store._io._env))
            del store
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_threaded_acquisition_opens_one_environment(self):
        original = self.store()
        original.write_eeg(np.arange(16), key='sample')

        def read_store(_):
            store = EEGSignalIO(self.directory.name, io_mode='lmdb')
            for _ in range(8):
                np.testing.assert_array_equal(store.read_eeg('sample'), np.arange(16))
            return store

        with ThreadPoolExecutor(max_workers=8) as executor:
            stores = list(executor.map(read_store, range(24)))
        self.assertTrue(all(store._io._env is original._io._env for store in stores))

    def test_map_growth_uses_shared_capacity_for_each_wrapper(self):
        first = self.store(io_size=32768)
        second = self.store(io_size=32768)
        first_sample = np.arange(250000, dtype=np.float32)
        second_sample = np.arange(500000, dtype=np.float32)
        first.write_eeg(first_sample, key='first')
        capacity = first._io._env.info()['map_size']
        second.write_eeg(second_sample, key='second')
        self.assertGreater(second._io._env.info()['map_size'], capacity)
        np.testing.assert_array_equal(first.read_eeg('second'), second_sample)
        np.testing.assert_array_equal(second.read_eeg('first'), first_sample)

    def test_failed_open_does_not_poison_a_later_valid_store(self):
        path = os.path.join(self.directory.name, 'store')
        with open(path, 'w') as file:
            file.write('not a directory')
        with self.assertRaises(OSError):
            EEGSignalIO(path, io_mode='lmdb')
        os.remove(path)
        store = self.store(path)
        store.write_eeg(np.arange(4), key='sample')
        np.testing.assert_array_equal(store.read_eeg('sample'), np.arange(4))

    def run_child(self, method, stale_capacity=False, hold_cache_lock=False):
        if stale_capacity:
            first = self.store(io_size=32768)
            store = self.store(io_size=32768)
            sample = np.arange(250000, dtype=np.float32)
            first.write_eeg(sample, key='sample')
        else:
            store = self.store()
            sample = np.arange(4)
            store.write_eeg(sample, key='sample')
        parent_environment = store._io._env
        context = multiprocessing.get_context(method)
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
            self.assertTrue(receiver.poll(30), 'Child did not return its sample.')
            result = receiver.recv()
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result[0], 'ok', result)
            self.assertEqual(result[1], sample.tolist())
            self.assertNotEqual(result[3], os.getpid())
            if method == 'fork':
                self.assertTrue(result[2], 'Child reused the inherited LMDB environment.')
            self.assertIs(store._io._env, parent_environment)
            np.testing.assert_array_equal(store.read_eeg('sample'), sample)
            store.write_eeg(np.arange(4) + 1, key='sample')
            np.testing.assert_array_equal(store.read_eeg('sample'), np.arange(4) + 1)
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            receiver.close()

    def test_spawned_process_reads_a_fresh_environment(self):
        self.run_child('spawn')

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'requires fork')
    def test_forked_process_reopens_inherited_environment(self):
        self.run_child('fork')

    def test_child_reopens_after_another_wrapper_grows_the_map(self):
        for method in ('spawn', 'fork'):
            if method in multiprocessing.get_all_start_methods():
                with self.subTest(method=method):
                    self.run_child(method, stale_capacity=True)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'requires fork')
    def test_forked_child_does_not_wait_on_inherited_cache_lock(self):
        self.run_child('fork', hold_cache_lock=True)

    def test_mne_dataset_cache_reuse_and_worker_loading(self):
        methods = [method for method in ('spawn', 'fork')
                   if method in multiprocessing.get_all_start_methods()]
        eeg = np.arange(800, dtype=float).reshape(2, 400)
        raw = mne.io.RawArray(eeg, mne.create_info(['C3', 'C4'], 100, 'eeg'), verbose='ERROR')
        expected = torch.tensor(np.stack([eeg[:, start:start + 100]
                                         for _ in range(2)
                                         for start in range(0, 400, 100)]), dtype=torch.float32)
        for threshold in (0, 128):
            for method in methods:
                with self.subTest(lazy_threshold=threshold, method=method):
                    parameters = dict(
                        raw_list=[raw, raw],
                        info_list=[{'subject_id': '01', 'trial_id': index} for index in range(2)],
                        chunk_size=100, io_mode='lmdb', lazy_threshold=threshold,
                        io_path=os.path.join(self.directory.name, f'dataset_{threshold}_{method}'),
                        online_transform=transforms.ToTensor(),
                        label_transform=transforms.Select('trial_id'), verbose=False)
                    dataset = MNERawDataset(**parameters)
                    reused = MNERawDataset(**parameters)
                    loader = DataLoader(reused, batch_size=2, num_workers=2,
                                        multiprocessing_context=method, timeout=30)
                    batches = list(loader)
                    torch.testing.assert_close(torch.cat([batch[0] for batch in batches]), expected)
                    torch.testing.assert_close(torch.cat([batch[1] for batch in batches]),
                                               torch.tensor([0] * 4 + [1] * 4))
                    np.testing.assert_array_equal(dataset[2][0].numpy(), eeg[:, 200:300])


if __name__ == '__main__':
    unittest.main()
