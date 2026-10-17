from copy import copy
import gc
import os
import pickle
from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BaseDataset, DEAPDataset, MNERawDataset
from torcheeg.model_selection import Subcategory


def add_one(data):
    return data + 1


def fail_hook(data):
    raise RuntimeError('hook failure')


class MemoryRecordsDataset(BaseDataset):

    @staticmethod
    def set_records(num_records=2, **kwargs):
        return list(range(num_records))

    @staticmethod
    def read_record(record, **kwargs):
        return {}

    @staticmethod
    def process_record(record, fail_record=None, reserved=False, **kwargs):
        if record == fail_record:
            raise RuntimeError('record failure')
        info = {'subject_id': record, 'session_id': 0, 'trial_id': 0}
        for index in range(3):
            key = f'{record}_{index}'
            info['clip_id'] = key
            if reserved:
                info['_record_id'] = 'forbidden'
            yield {'key': key, 'eeg': np.full((2, 4), record * 10 + index, dtype=np.float32),
                   'info': info}


class TestMemoryDatasetStorage(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def assert_custom_samples(self, dataset, increment=0):
        self.assertFalse(dataset.is_lazy())
        self.assertFalse(getattr(dataset, 'eeg_signal_paths', {}))
        self.assertEqual(len(dataset), 6)
        self.assertEqual(len(set(dataset.info.clip_id)), 6)
        for index in range(6):
            signal, info = dataset[index]
            record, window = index // 3, index % 3
            np.testing.assert_array_equal(signal, np.full((2, 4), record * 10 + window + increment))
            self.assertEqual(info['clip_id'], f'{record}_{window}')
            self.assertEqual(info['subject_id'], record)

    def test_memory_records_work_without_a_disk_path(self):
        dataset = MemoryRecordsDataset(io_mode='memory', io_path=None, verbose=False)
        self.assert_custom_samples(dataset)

    def test_memory_storage_is_retained_with_sequential_and_parallel_preprocessing(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                cache = self.path / f'cache_{workers}'
                dataset = MemoryRecordsDataset(io_mode='memory', io_path=str(cache),
                                               num_worker=workers, lazy_threshold=0, verbose=False)
                self.assert_custom_samples(dataset)
                self.assertFalse(cache.exists())

    def test_mne_raw_windows_and_transforms_are_retained(self):
        eeg = np.arange(800, dtype=float).reshape(2, 400)
        raw = mne.io.RawArray(eeg, mne.create_info(['C3', 'C4'], 100, 'eeg'), verbose='ERROR')
        for workers in (0, 2):
            with self.subTest(workers=workers):
                dataset = MNERawDataset(
                    raw_list=[raw], info_list=[{'subject_id': 1, 'trial_id': 0}],
                    chunk_size=100, io_mode='memory', io_path=str(self.path / f'mne_{workers}'),
                    offline_transform=transforms.Lambda(add_one),
                    num_worker=workers, lazy_threshold=0, verbose=False)
                self.assertEqual(len(dataset), 4)
                self.assertFalse(dataset.is_lazy())
                for index in range(4):
                    np.testing.assert_array_equal(dataset[index][0], eeg[:, index * 100:(index + 1) * 100] + 1)

    def test_dataset_copies_keep_samples_and_independent_metadata(self):
        dataset = MemoryRecordsDataset(io_mode='memory', io_path=None, verbose=False)
        duplicate = copy(dataset)
        duplicate.info.loc[0, 'trial_id'] = 9
        self.assertEqual(dataset.info.loc[0, 'trial_id'], 0)
        del dataset
        gc.collect()
        self.assert_custom_samples(duplicate)
        replacement = np.full((2, 4), 99.)
        second = copy(duplicate)
        second.write_eeg('_record_0', '0_0', replacement)
        np.testing.assert_array_equal(duplicate[0][0], replacement)

    def test_subcategory_reads_memory_views(self):
        dataset = MemoryRecordsDataset(io_mode='memory', io_path=None, verbose=False)
        splitter = Subcategory(criteria='subject_id', split_path=str(self.path / 'splits'))
        subsets = list(splitter.split(dataset))
        self.assertEqual(len(subsets), 2)
        for record, subset in enumerate(subsets):
            self.assertEqual(len(subset), 3)
            for index in range(3):
                signal, info = subset[index]
                np.testing.assert_array_equal(signal, np.full((2, 4), record * 10 + index))
                self.assertEqual(info['subject_id'], record)

    def test_deap_baseline_keys_survive_memory_preprocessing(self):
        root = self.path / 'deap'
        root.mkdir()
        data = np.arange(2 * 32 * 12, dtype=float).reshape(2, 32, 12)
        labels = np.array([[3., 4., 5., 6.], [7., 8., 2., 3.]])
        with (root / 's01.dat').open('wb') as file:
            pickle.dump({'data': data, 'labels': labels}, file)
        dataset = DEAPDataset(
            root_path=str(root), io_mode='memory', io_path=str(self.path / 'deap-cache'),
            chunk_size=4, baseline_chunk_size=4, num_baseline=1,
            offline_transform=transforms.Lambda(add_one),
            online_transform=transforms.BaselineRemoval(),
            label_transform=transforms.Select('valence'), verbose=False)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.eeg_io_router['_record_0']), 6)
        for index in range(4):
            signal, label = dataset[index]
            np.testing.assert_array_equal(signal, np.full((32, 4), 4 * (index % 2 + 1)))
            self.assertEqual(label, labels[index // 2, 0])

    def test_all_after_hooks_update_retained_samples(self):
        dataset = MemoryRecordsDataset(
            io_mode='memory', io_path=None, after_trial=add_one,
            after_session=add_one, after_subject=add_one, verbose=False)
        self.assert_custom_samples(dataset, increment=3)

    def test_repeated_construction_does_not_modify_an_existing_directory(self):
        cache = self.path / 'existing'
        cache.mkdir()
        sentinel = cache / 'keep.bin'
        sentinel.write_bytes(b'existing cached data')
        first = MemoryRecordsDataset(io_mode='memory', io_path=str(cache), verbose=False)
        second = MemoryRecordsDataset(io_mode='memory', io_path=str(cache), verbose=False)
        self.assert_custom_samples(first)
        self.assert_custom_samples(second)
        first.write_eeg('_record_0', '0_0', np.full((2, 4), 99.))
        np.testing.assert_array_equal(second[0][0], np.zeros((2, 4)))
        self.assertEqual(sentinel.read_bytes(), b'existing cached data')
        self.assertEqual(set(os.listdir(cache)), {'keep.bin'})

    def test_failures_preserve_existing_files(self):
        for failure in ('record', 'hook'):
            with self.subTest(failure=failure):
                cache = self.path / failure
                cache.mkdir()
                sentinel = cache / 'keep.bin'
                sentinel.write_bytes(b'preserved')
                kwargs = {'fail_record': 1} if failure == 'record' else {'after_trial': fail_hook}
                with self.assertRaisesRegex(RuntimeError, failure + ' failure'):
                    MemoryRecordsDataset(io_mode='memory', io_path=str(cache), verbose=False, **kwargs)
                self.assertEqual(sentinel.read_bytes(), b'preserved')
                self.assertEqual(set(os.listdir(cache)), {'keep.bin'})

    def test_spawned_loader_reads_retained_memory_samples(self):
        dataset = MemoryRecordsDataset(io_mode='memory', io_path=None, verbose=False)
        batches = list(DataLoader(dataset, batch_size=2, num_workers=2,
                                  multiprocessing_context='spawn', timeout=30))
        expected = np.stack([np.full((2, 4), record * 10 + index, dtype=np.float32)
                             for record in range(2) for index in range(3)])
        torch.testing.assert_close(torch.cat([batch[0] for batch in batches]), torch.from_numpy(expected))

    def test_reserved_metadata_column_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'reserved'):
            MemoryRecordsDataset(io_mode='memory', io_path=str(self.path / 'reserved'),
                                 reserved=True, verbose=False)

    def test_empty_record_list_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'No records'):
            MemoryRecordsDataset(io_mode='memory', io_path=None, num_records=0, verbose=False)


if __name__ == '__main__':
    unittest.main()
