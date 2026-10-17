import os
import tempfile
import unittest
from copy import copy

import numpy as np

from torcheeg.io import EEGSignalIO


class TestAutomaticSignalKeys(unittest.TestCase):
    modes = ('memory', 'pickle', 'lmdb')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.counter = 0

    def make_store(self, mode, path=None, io_size=1048576):
        self.counter += 1
        if path is None:
            path = os.path.join(self.directory.name, f'{mode}_{self.counter}')
        store = EEGSignalIO(path, io_mode=mode, io_size=io_size)
        if mode == 'lmdb':
            self.addCleanup(store._io._env.close)
        return store

    def assert_samples(self, store, expected):
        self.assertEqual(set(store.keys()), set(expected))
        self.assertEqual(len(store), len(expected))
        for key, sample in expected.items():
            np.testing.assert_array_equal(store.read_eeg(key), sample)

    def test_ordinary_automatic_keys_remain_sequential(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                expected = {}
                for index in range(3):
                    sample = np.full((2, 4), index, dtype=np.float32)
                    key = store.write_eeg(sample)
                    self.assertEqual(key, str(index))
                    expected[key] = sample
                self.assert_samples(store, expected)

    def test_sparse_explicit_keys_are_not_overwritten(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                expected = {'0': np.array([10]), '2': np.array([20])}
                for key, sample in expected.items():
                    store.write_eeg(sample, key=key)
                self.assertEqual(store._io.write_pointer, 3)
                for key, value in (('3', 30), ('4', 40)):
                    sample = np.array([value])
                    self.assertEqual(store.write_eeg(sample), key)
                    expected[key] = sample
                self.assert_samples(store, expected)

    def test_search_skips_multiple_occupied_integer_keys(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                expected = {str(index): np.array([index]) for index in (3, 4, 5)}
                for key, sample in expected.items():
                    store.write_eeg(sample, key=key)
                self.assertEqual(store.write_eeg(np.array([99])), '6')
                expected['6'] = np.array([99])
                self.assert_samples(store, expected)

    def test_noncolliding_text_and_numeric_keys_keep_existing_behavior(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                expected = {key: np.array([index]) for index, key in
                            enumerate(('100', 'record_0', '00', '\u00b2'))}
                for key, sample in expected.items():
                    store.write_eeg(sample, key=key)
                self.assertEqual(store.write_eeg(np.array([99])), '4')
                expected['4'] = np.array([99])
                self.assert_samples(store, expected)

    def test_explicit_replacement_and_failed_none_writes(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                store.write_eeg(np.array([10]), key='0')
                store.write_eeg(np.array([20]), key='2')
                self.assertEqual(store.write_eeg(np.array([22]), key='2'), '2')
                for key in (None, '2'):
                    with self.assertRaisesRegex(RuntimeError, 'Save None'):
                        store.write_eeg(None, key=key)
                self.assert_samples(store, {'0': np.array([10]), '2': np.array([22])})
                self.assertEqual(store.write_eeg(np.array([30])), '3')
                self.assert_samples(store, {'0': np.array([10]), '2': np.array([22]),
                                            '3': np.array([30])})

    def test_copied_wrappers_append_without_losing_either_view(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                store.write_eeg(np.array([10]), key='0')
                store.write_eeg(np.array([20]), key='2')
                duplicate = copy(store)
                self.assertEqual(duplicate.write_eeg(np.array([30])), '3')
                self.assertEqual(store.write_eeg(np.array([40])), '4')
                expected = {str(index): np.array([10 * index]) for index in (2, 3, 4)}
                expected['0'] = np.array([10])
                self.assert_samples(store, expected)
                self.assert_samples(duplicate, expected)

    def test_reopened_stores_use_the_persisted_keys(self):
        for mode in ('pickle', 'lmdb'):
            with self.subTest(mode=mode):
                store = self.make_store(mode)
                store.write_eeg(np.array([10]), key='0')
                store.write_eeg(np.array([20]), key='2')
                path = store.io_path
                if mode == 'lmdb':
                    store._io._env.close()
                del store
                restored = self.make_store(mode, path=path)
                self.assertEqual(restored.write_eeg(np.array([30])), '3')
                self.assert_samples(restored, {'0': np.array([10]), '2': np.array([20]),
                                               '3': np.array([30])})

    def test_mode_conversions_preserve_collision_free_append(self):
        store = self.make_store('memory')
        store.write_eeg(np.array([10]), key='0')
        store.write_eeg(np.array([20]), key='2')
        store.to_pickle(os.path.join(self.directory.name, 'converted_pickle'))
        self.assertEqual(store.write_eeg(np.array([30])), '3')
        store.to_lmdb(os.path.join(self.directory.name, 'converted_lmdb'))
        self.addCleanup(store._io._env.close)
        self.assertEqual(store.write_eeg(np.array([40])), '4')
        self.assert_samples(store, {'0': np.array([10]), '2': np.array([20]),
                                    '3': np.array([30]), '4': np.array([40])})

    def test_lmdb_map_growth_retries_an_automatic_key_safely(self):
        store = self.make_store('lmdb', io_size=32768)
        store.write_eeg(np.array([10]), key='0')
        store.write_eeg(np.array([20]), key='2')
        sample = np.arange(250000, dtype=np.float32)
        self.assertEqual(store.write_eeg(sample), '3')
        self.assertGreater(store._io.io_size, 32768)
        self.assert_samples(store, {'0': np.array([10]), '2': np.array([20]), '3': sample})


if __name__ == '__main__':
    unittest.main()
