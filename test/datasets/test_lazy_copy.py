from copy import copy
import gc
import multiprocessing
import os
import tempfile
import unittest

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg.datasets import MNERawDataset
from torcheeg.model_selection import KFoldCrossSubject, Subcategory, train_test_split


class TestLazyDatasetCopy(unittest.TestCase):
    modes = ('pickle', 'lmdb')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.counter = 0
        self.eeg = np.arange(800, dtype=float).reshape(2, 400)

    def make_dataset(self, mode):
        self.counter += 1
        raw_list = [mne.io.RawArray(self.eeg + 1000 * subject,
                                   mne.create_info(['C3', 'C4'], 100, 'eeg'),
                                   verbose='ERROR') for subject in (1, 2)]
        return MNERawDataset(
            raw_list=raw_list,
            info_list=[{'subject_id': subject, 'trial_id': 0} for subject in (1, 2)],
            chunk_size=100, io_mode=mode, lazy_threshold=0,
            io_path=os.path.join(self.directory.name, f'cache_{self.counter}'), verbose=False)

    def assert_samples(self, dataset):
        self.assertTrue(dataset.is_lazy())
        self.assertEqual(dataset.eeg_io_router, {})
        for index in range(len(dataset)):
            signal, info = dataset[index]
            start = int(info['clip_id'].split('_')[-1])
            expected = self.eeg[:, start:start + 100] + 1000 * info['subject_id']
            np.testing.assert_array_equal(signal, expected)

    def test_copy_preserves_lazy_reads_after_original_is_deleted(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                duplicate = copy(dataset)
                self.assertEqual(duplicate.io_path, dataset.io_path)
                self.assertEqual(duplicate.io_mode, mode)
                del dataset
                gc.collect()
                self.assert_samples(duplicate)

    def test_metadata_and_path_dictionaries_are_independent(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                duplicate = copy(dataset)
                self.assertIsNot(duplicate.info, dataset.info)
                self.assertIsNot(duplicate.eeg_signal_paths, dataset.eeg_signal_paths)
                signal, _ = dataset[0]
                duplicate.info.loc[0, 'subject_id'] = 99
                duplicate.eeg_signal_paths['unused'] = 'not opened'
                actual, info = duplicate[0]
                np.testing.assert_array_equal(actual, signal)
                self.assertEqual(info['subject_id'], 99)
                self.assertEqual(dataset.info.loc[0, 'subject_id'], 1)
                self.assertNotIn('unused', dataset.eeg_signal_paths)

    def test_subcategory_partitions_read_all_samples(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                splitter = Subcategory(criteria='subject_id', split_path=dataset.io_path + '-categories')
                subsets = list(splitter.split(dataset))
                self.assertEqual(len(subsets), 2)
                self.assertEqual([set(subset.info.subject_id) for subset in subsets], [{1}, {2}])
                for subset in subsets:
                    self.assertEqual(len(subset), 4)
                    self.assert_samples(subset)
                for subset in splitter.split(dataset):
                    self.assert_samples(subset)

    def test_subject_folds_preserve_lazy_loading_and_isolation(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                splitter = KFoldCrossSubject(n_splits=2, split_path=dataset.io_path + '-folds')
                folds = list(splitter.split(dataset))
                self.assertEqual(len(folds), 2)
                for train, test in folds:
                    self.assertFalse(set(train.info.subject_id) & set(test.info.subject_id))
                    self.assertEqual(set(train.info.clip_id) | set(test.info.clip_id), set(dataset.info.clip_id))
                    self.assert_samples(train)
                    self.assert_samples(test)

    def test_sample_split_reads_each_cached_row(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                train, test = train_test_split(dataset, test_size=0.25, random_state=42,
                                              split_path=dataset.io_path + '-samples')
                self.assertEqual((len(train), len(test)), (6, 2))
                self.assertFalse(set(train.info.clip_id) & set(test.info.clip_id))
                self.assertEqual(set(train.info.clip_id) | set(test.info.clip_id), set(dataset.info.clip_id))
                self.assert_samples(train)
                self.assert_samples(test)

    def test_copy_keeps_the_existing_shared_storage_contract(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                dataset = self.make_dataset(mode)
                duplicate = copy(dataset)
                info = duplicate.read_info(0)
                replacement = np.full((2, 100), 7.)
                duplicate.write_eeg(info['_record_id'], info['clip_id'], replacement)
                np.testing.assert_array_equal(dataset[0][0], replacement)
                np.testing.assert_array_equal(duplicate[0][0], replacement)

    def test_spawned_loader_reads_a_copied_lazy_dataset(self):
        if 'spawn' not in multiprocessing.get_all_start_methods():
            self.skipTest('requires spawn')
        expected = np.stack([self.eeg[:, start:start + 100] + 1000 * subject
                             for subject in (1, 2) for start in range(0, 400, 100)])
        for mode in self.modes:
            with self.subTest(mode=mode):
                duplicate = copy(self.make_dataset(mode))
                batches = list(DataLoader(duplicate, batch_size=2, num_workers=2,
                                         multiprocessing_context='spawn', timeout=30))
                torch.testing.assert_close(torch.cat([batch[0] for batch in batches]),
                                           torch.from_numpy(expected))
                self.assert_samples(duplicate)


if __name__ == '__main__':
    unittest.main()
