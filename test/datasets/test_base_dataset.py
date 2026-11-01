import gc
import json
import multiprocessing
import os
import tempfile
import unittest
from copy import copy
from pathlib import Path

import mne
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BaseDataset, MNERawDataset
from torcheeg.model_selection import (KFoldCrossSubject, Subcategory,
                                      train_test_split)


class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_missing_record_hooks_identify_the_subclass(self):
        class IncompleteDataset(BaseDataset):
            pass

        dataset = IncompleteDataset.__new__(IncompleteDataset)
        for method, kwargs in (("set_records", {}),
                               ("read_record", {"record": "record"}),
                               ("process_record", {"record": "record"})):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                        NotImplementedError,
                        f"Method {method} is not implemented in class IncompleteDataset"):
                    getattr(dataset, method)(**kwargs)

    def test_record_hooks_can_be_overridden_as_static_methods(self):
        class StaticDataset(BaseDataset):
            @staticmethod
            def read_record(record, **kwargs):
                return {"samples": record}

            @staticmethod
            def process_record(record, samples, **kwargs):
                yield {"eeg": samples, "key": record}

        dataset = StaticDataset.__new__(StaticDataset)
        for owner in (StaticDataset, dataset):
            samples = owner.read_record(record="record")
            self.assertEqual(next(owner.process_record(record="record", **samples)),
                             {"eeg": "record", "key": "record"})


class _MemoryDataset(BaseDataset):
    @staticmethod
    def set_records(num_records=2, **kwargs):
        return list(range(num_records))

    @staticmethod
    def read_record(record, **kwargs):
        return {}

    @staticmethod
    def process_record(record, reserved=False, fail_record=None, **kwargs):
        if record == fail_record:
            raise RuntimeError('record failure')
        baseline_id = f'09{record:02}'
        yield {'key': baseline_id,
               'eeg': np.full((2, 4), -record - 1, dtype=np.float32)}
        info = {'subject_id': f'{record + 1:03}', 'session_id': '000',
                'trial_id': '000', 'baseline_id': baseline_id, 'tag': 'NA'}
        if reserved:
            info['_record_id'] = 'forbidden'
        for index in range(2):
            key = f'{record:02}{index:02}'
            info['clip_id'] = key
            yield {'key': key,
                   'eeg': np.full((2, 4), record * 10 + index, dtype=np.float32),
                   'info': info}


def _increment_memory_samples(data):
    return data + 1


class TestMemoryDataset(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def _assert_samples(self, dataset, increment=0):
        self.assertEqual(len(dataset), 4)
        self.assertFalse(dataset.is_lazy())
        self.assertEqual(dataset.eeg_signal_paths, {})
        for index in range(4):
            record, window = divmod(index, 2)
            signal, info = dataset[index]
            self.assertEqual(info['clip_id'], f'{record:02}{window:02}')
            self.assertEqual(info['subject_id'], f'{record + 1:03}')
            self.assertEqual(info['session_id'], '000')
            self.assertEqual(info['trial_id'], '000')
            self.assertEqual(info['baseline_id'], f'09{record:02}')
            self.assertEqual(info['tag'], 'NA')
            np.testing.assert_array_equal(signal, np.full((2, 4),
                                          record * 10 + window + increment))
            np.testing.assert_array_equal(dataset.read_eeg(
                info['_record_id'], info['baseline_id']),
                np.full((2, 4), -record - 1))

    def test_string_ids_survive_sequential_and_parallel_preprocessing(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                dataset = _MemoryDataset(io_mode='memory', num_worker=workers,
                                         lazy_threshold=0, verbose=False)
                self._assert_samples(dataset)

    def test_memory_storage_does_not_use_disk_metadata_or_reuse_a_cache(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'existing-file')
            with open(path, 'wb') as file:
                file.write(b'preserve')
            with patch('torcheeg.datasets.base_dataset.MetaInfoIO',
                       side_effect=AssertionError('memory mode must not use CSV')):
                first = _MemoryDataset(
                    io_mode='memory', io_path=path, verbose=False)
                second = _MemoryDataset(
                    io_mode='memory', io_path=path, verbose=False)
            self._assert_samples(first)
            self._assert_samples(second)
            first.write_eeg('_record_0', '0000', np.full((2, 4), 99))
            np.testing.assert_array_equal(second[0][0], np.zeros((2, 4)))
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'preserve')
            self.assertEqual(os.listdir(directory), ['existing-file'])
            with self.assertRaisesRegex(RuntimeError, 'record failure'):
                _MemoryDataset(io_mode='memory', io_path=path,
                               fail_record=1, verbose=False)
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'preserve')

    def test_after_hooks_use_preserved_string_ids(self):
        dataset = _MemoryDataset(io_mode='memory', verbose=False,
                                 after_trial=_increment_memory_samples,
                                 after_session=_increment_memory_samples,
                                 after_subject=_increment_memory_samples)
        self._assert_samples(dataset, increment=3)

    def test_copied_and_partitioned_views_preserve_string_ids(self):
        dataset = _MemoryDataset(io_mode='memory', verbose=False)
        duplicate = copy(dataset)
        duplicate.info.loc[0, 'tag'] = 'changed'
        self.assertEqual(dataset.info.loc[0, 'tag'], 'NA')
        np.testing.assert_array_equal(duplicate[0][0], dataset[0][0])
        with tempfile.TemporaryDirectory() as directory:
            subsets = list(Subcategory(criteria='subject_id',
                                       split_path=os.path.join(directory, 'split')).split(dataset))
        self.assertEqual(len(subsets), 2)
        for subject, subset in zip(('001', '002'), subsets):
            self.assertEqual(set(subset.info['subject_id']), {subject})
            self.assertEqual(len(subset), 2)
            for index in range(2):
                signal, info = subset[index]
                self.assertEqual(info['subject_id'], subject)
                self.assertEqual(signal.shape, (2, 4))

    def test_spawned_loader_reads_samples_by_original_string_ids(self):
        dataset = _MemoryDataset(io_mode='memory', verbose=False)
        batches = list(DataLoader(dataset, batch_size=2, num_workers=2,
                                  multiprocessing_context='spawn', timeout=30))
        expected = np.stack([np.full((2, 4), record * 10 + index, dtype=np.float32)
                             for record in range(2) for index in range(2)])
        torch.testing.assert_close(torch.cat([batch[0] for batch in batches]),
                                   torch.from_numpy(expected))
        self.assertEqual([key for _, info in batches for key in info['clip_id']],
                         ['0000', '0001', '0100', '0101'])

    def test_memory_initialization_retains_reserved_column_and_empty_record_checks(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            _MemoryDataset(io_mode='memory', reserved=True, verbose=False)
        with self.assertRaisesRegex(ValueError, 'No records'):
            _MemoryDataset(io_mode='memory', num_records=0, verbose=False)


class TestLazyDatasetCopy(unittest.TestCase):
    _modes = ('pickle', 'lmdb')

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._counter = 0
        self._eeg = np.arange(800, dtype=float).reshape(2, 400)

    def _make_dataset(self, mode):
        self._counter += 1
        raw_list = [mne.io.RawArray(self._eeg + 1000 * subject,
                                    mne.create_info(['C3', 'C4'], 100, 'eeg'),
                                    verbose='ERROR') for subject in (1, 2)]
        return MNERawDataset(
            raw_list=raw_list,
            info_list=[{'subject_id': subject, 'trial_id': 0}
                       for subject in (1, 2)],
            chunk_size=100, io_mode=mode, lazy_threshold=0,
            io_path=os.path.join(self._directory.name, f'cache_{self._counter}'), verbose=False)

    def _assert_samples(self, dataset):
        self.assertTrue(dataset.is_lazy())
        self.assertEqual(dataset.eeg_io_router, {})
        for index in range(len(dataset)):
            signal, info = dataset[index]
            start = int(info['clip_id'].split('_')[-1])
            expected = self._eeg[:, start:start + 100] + \
                1000 * info['subject_id']
            np.testing.assert_array_equal(signal, expected)

    def test_copy_preserves_lazy_reads_after_original_is_deleted(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                self.assertEqual(duplicate.io_path, dataset.io_path)
                self.assertEqual(duplicate.io_mode, mode)
                del dataset
                gc.collect()
                self._assert_samples(duplicate)

    def test_metadata_and_path_dictionaries_are_independent(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                self.assertIsNot(duplicate.info, dataset.info)
                self.assertIsNot(duplicate.eeg_signal_paths,
                                 dataset.eeg_signal_paths)
                signal, _ = dataset[0]
                duplicate.info.loc[0, 'subject_id'] = 99
                duplicate.eeg_signal_paths['unused'] = 'not opened'
                actual, info = duplicate[0]
                np.testing.assert_array_equal(actual, signal)
                self.assertEqual(info['subject_id'], 99)
                self.assertEqual(dataset.info.loc[0, 'subject_id'], 1)
                self.assertNotIn('unused', dataset.eeg_signal_paths)

    def test_subcategory_partitions_read_all_samples(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                splitter = Subcategory(
                    criteria='subject_id', split_path=dataset.io_path + '-categories')
                subsets = list(splitter.split(dataset))
                self.assertEqual(len(subsets), 2)
                self.assertEqual([set(subset.info.subject_id)
                                 for subset in subsets], [{1}, {2}])
                for subset in subsets:
                    self.assertEqual(len(subset), 4)
                    self._assert_samples(subset)
                for subset in splitter.split(dataset):
                    self._assert_samples(subset)

    def test_subject_folds_preserve_lazy_loading_and_isolation(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                splitter = KFoldCrossSubject(
                    n_splits=2, split_path=dataset.io_path + '-folds')
                folds = list(splitter.split(dataset))
                self.assertEqual(len(folds), 2)
                for train, test in folds:
                    self.assertFalse(set(train.info.subject_id)
                                     & set(test.info.subject_id))
                    self.assertEqual(set(train.info.clip_id) | set(
                        test.info.clip_id), set(dataset.info.clip_id))
                    self._assert_samples(train)
                    self._assert_samples(test)

    def test_sample_split_reads_each_cached_row(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                train, test = train_test_split(dataset, test_size=0.25, random_state=42,
                                               split_path=dataset.io_path + '-samples')
                self.assertEqual((len(train), len(test)), (6, 2))
                self.assertFalse(set(train.info.clip_id)
                                 & set(test.info.clip_id))
                self.assertEqual(set(train.info.clip_id) | set(
                    test.info.clip_id), set(dataset.info.clip_id))
                self._assert_samples(train)
                self._assert_samples(test)

    def test_copy_keeps_the_existing_shared_storage_contract(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                info = duplicate.read_info(0)
                replacement = np.full((2, 100), 7.)
                duplicate.write_eeg(
                    info['_record_id'], info['clip_id'], replacement)
                np.testing.assert_array_equal(dataset[0][0], replacement)
                np.testing.assert_array_equal(duplicate[0][0], replacement)

    def test_spawned_loader_reads_a_copied_lazy_dataset(self):
        if 'spawn' not in multiprocessing.get_all_start_methods():
            self.skipTest('requires spawn')
        expected = np.stack([self._eeg[:, start:start + 100] + 1000 * subject
                             for subject in (1, 2) for start in range(0, 400, 100)])
        for mode in self._modes:
            with self.subTest(mode=mode):
                duplicate = copy(self._make_dataset(mode))
                batches = list(DataLoader(duplicate, batch_size=2, num_workers=2,
                                          multiprocessing_context='spawn', timeout=30))
                torch.testing.assert_close(torch.cat([batch[0] for batch in batches]),
                                           torch.from_numpy(expected))
                self._assert_samples(duplicate)


class TestDatasetMetadata(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_missing_metadata_raises_on_class_access(self):
        for name in ('channels', 'grid', 'adjacency'):
            with self.subTest(attribute=name):
                with self.assertRaisesRegex(
                        NotImplementedError,
                        f'Attribute {name} is not implemented in class BaseDataset'):
                    getattr(BaseDataset, name)


class _CachedDataset(BaseDataset):

    def set_records(self, root_path, **kwargs):
        return sorted(str(path) for path in Path(root_path).glob('*.npy'))

    @staticmethod
    def read_record(record, **kwargs):
        return {'samples': np.load(record)}

    @staticmethod
    def process_record(record,
                       samples,
                       scale=1,
                       offline_transform=None,
                       fail=False,
                       **kwargs):
        if fail:
            raise RuntimeError('Interrupted writer')
        for index, sample in enumerate(samples):
            eeg = sample * scale
            if offline_transform is not None:
                eeg = offline_transform(eeg=eeg)['eeg']
            yield {
                'eeg': eeg,
                'key': str(index),
                'info': {
                    'clip_id': str(index),
                    'subject_id': 0,
                    'trial_id': 0
                }
            }


def _dataset_worker(source, path, queue):
    try:
        dataset = _CachedDataset(root_path=source,
                                 io_path=path,
                                 io_mode='pickle',
                                 verbose=False)
        queue.put((len(dataset), float(dataset[0][0].sum())))
    except Exception as error:
        queue.put(repr(error))


def _add_one(samples):
    return samples + 1


def _fail_postprocessing(samples):
    raise RuntimeError('Interrupted postprocessing')


class TestDatasetCache(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._root = Path(self._directory.name)
        self._source = self._root / 'source'
        self._source.mkdir()
        self._samples = np.arange(24).reshape(3, 2, 4)
        np.save(self._source / 'record.npy', self._samples)
        self._path = self._root / 'cache'

    def _dataset(self, **kwargs):
        parameters = dict(root_path=str(self._source),
                          io_path=str(self._path),
                          io_mode='pickle',
                          verbose=False)
        parameters.update(kwargs)
        return _CachedDataset(**parameters)

    def test_disk_modes_preserve_samples_and_reuse(self):
        for mode in ('pickle', 'lmdb'):
            with self.subTest(mode=mode):
                path = self._root / mode
                first = self._dataset(io_path=path,
                                      io_mode=mode,
                                      lazy_threshold=0,
                                      after_trial=_add_one)
                np.testing.assert_array_equal(
                    first[0][0], self._samples[0] + 1)
                manifest = path / 'manifest.json'
                before = manifest.read_bytes()
                repeated = self._dataset(
                    io_path=path,
                    io_mode=mode,
                    lazy_threshold=0,
                    after_trial=_add_one,
                    num_worker=1,
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Select('subject_id'))
                np.testing.assert_array_equal(
                    repeated[0][0], self._samples[0] + 1)
                self.assertEqual(before, manifest.read_bytes())
                self.assertEqual(set(json.loads(before)),
                                 {'fingerprint', 'files'})

    def test_offline_parameters_and_hooks_invalidate_cache(self):
        self._dataset()
        for kwargs in ({
                'scale': 2
        }, {
                'offline_transform': transforms.MeanStdNormalize()
        }, {
                'after_trial': _add_one
        }):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, 'changed'):
                self._dataset(**kwargs)

    def test_lambda_transform_configuration_is_stable(self):

        def make_transform(offset):
            return lambda eeg: {'eeg': eeg + offset}

        first = self._dataset(offline_transform=make_transform(1))
        repeated = self._dataset(offline_transform=make_transform(1))
        np.testing.assert_array_equal(first[0][0], repeated[0][0])
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset(offline_transform=make_transform(2))

    def test_source_content_is_checked_even_if_size_and_time_match(self):
        self._dataset()
        file = self._source / 'record.npy'
        stat = file.stat()
        np.save(file, self._samples + 1)
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset()

    def test_changed_or_missing_outputs_are_rejected(self):
        self._dataset()
        sample = next((self._path / '_record_0' / 'eeg').iterdir())
        original = sample.read_bytes()
        sample.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            self._dataset()
        sample.write_bytes(original)
        sample.unlink()
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset()

    def test_failed_writer_and_postprocessing_can_retry(self):
        for kwargs, message in (({
                'fail': True
        }, 'Interrupted writer'), ({
                'after_trial': _fail_postprocessing
        }, 'Interrupted postprocessing')):
            with self.subTest(kwargs=kwargs):
                path = self._root / message.replace(' ', '_')
                with self.assertRaisesRegex(RuntimeError, message):
                    self._dataset(io_path=path, **kwargs)
                self.assertFalse(path.exists())
                self.assertFalse(list(self._root.glob('.torcheeg-cache-*')))
                recovered = self._dataset(io_path=path)
                np.testing.assert_array_equal(
                    recovered[0][0], self._samples[0])

    def test_unknown_existing_cache_is_preserved(self):
        self._path.mkdir()
        file = self._path / 'old.csv'
        file.write_text('old data')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self._dataset()
        self.assertEqual(file.read_text(), 'old data')

    def test_cache_waits_for_another_writer(self):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        from threading import Event

        started = Event()

        def create_dataset():
            started.set()
            return self._dataset()

        with ThreadPoolExecutor(max_workers=1) as executor:
            with FileLock(str(self._path) + '.lock'):
                future = executor.submit(create_dataset)
                self.assertTrue(started.wait(timeout=10))
                with self.assertRaises(TimeoutError):
                    future.result(timeout=0.1)
            dataset = future.result(timeout=20)
        np.testing.assert_array_equal(dataset[0][0], self._samples[0])

    def test_competing_processes_publish_one_complete_cache(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        processes = [
            context.Process(target=_dataset_worker,
                            args=(str(self._source), str(self._path), queue))
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=60) for _ in processes]
            self.assertEqual(results, [(3, float(self._samples[0].sum()))] * 2)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            queue.close()
            queue.join_thread()

    def test_memory_mode_retains_signals_without_persistent_manifest(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                dataset = self._dataset(io_mode='memory',
                                        num_worker=workers,
                                        after_trial=_add_one)
                np.testing.assert_array_equal(
                    dataset[1][0], self._samples[1] + 1)
                self.assertFalse(self._path.exists())

    def test_omitted_paths_create_separate_caches(self):
        first = self._dataset(io_path=None)
        second = self._dataset(io_path=None)
        self.assertNotEqual(first.io_path, second.io_path)
        # BaseDataset defaults are relative to the working directory.
        import shutil
        self.addCleanup(shutil.rmtree, first.io_path)
        self.addCleanup(shutil.rmtree, second.io_path)
        self.assertTrue(Path(first.io_path, 'manifest.json').is_file())
        self.assertTrue(Path(second.io_path, 'manifest.json').is_file())
        self.addCleanup(Path(first.io_path + '.lock').unlink, missing_ok=True)
        self.addCleanup(Path(second.io_path + '.lock').unlink, missing_ok=True)


if __name__ == '__main__':
    unittest.main()
