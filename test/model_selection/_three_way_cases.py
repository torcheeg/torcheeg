import concurrent.futures
import multiprocessing as mp
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from torcheeg import model_selection
from torcheeg.datasets import BaseDataset
from torcheeg.io import EEGSignalIO


def make_three_way_dataset():
    dataset = BaseDataset.__new__(BaseDataset)
    rows = []
    io = EEGSignalIO('unused', io_mode='memory')
    for subject in range(5):
        for trial in range(5):
            for clip in range(5):
                index = len(rows)
                io.write_eeg(np.array([index, trial, clip]), key=str(index))
                rows.append({'subject_id': subject, 'trial_id': trial,
                             'clip_id': str(index), '_record_id': '_record_0',
                             'split': ('train', 'val', 'test')[index % 3]})
    dataset.info = pd.DataFrame(rows)
    dataset.info['category'] = pd.Categorical(['rest', 'task', 'task', 'rest', 'rest'] * 25)
    dataset.info['timestamp'] = pd.date_range('2020-01-01', periods=len(rows))
    dataset.eeg_io_router = {'_record_0': io}
    dataset.eeg_signal_paths = {}
    return dataset


def split_worker(class_name, parameters, split_path, queue):
    try:
        splitter = getattr(model_selection, class_name)(split_path=split_path, **parameters)
        folds = list(splitter.split(make_three_way_dataset()))
        queue.put(('ok', [[list(subset.info['clip_id']) for subset in fold]
                           for fold in folds]))
    except Exception as error:
        queue.put(('error', repr(error)))


class ThreeWayChecks:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.split_path = os.path.join(self.directory.name, 'split')

    def make_splitter(self, split_path=None, **overrides):
        parameters = dict(self.parameters)
        parameters.update(overrides)
        return self.splitter_class(split_path=self.split_path if split_path is None
                                   else split_path, **parameters)

    def assert_partition(self, dataset, folds):
        self.assertEqual(len(folds), self.fold_count)
        for fold in folds:
            self.assertEqual(len(fold), 3)
            clips = [set(subset.info['clip_id']) for subset in fold]
            self.assertEqual(set.union(*clips), set(dataset.info['clip_id']))
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(clips[first] & clips[second])
                if self.group_column is not None:
                    self.assertFalse(set(fold[first].info[self.group_column]) &
                                     set(fold[second].info[self.group_column]))
            for subset in fold:
                self.assertGreater(len(subset), 0)
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))

    def test_samples_and_groups_form_complete_disjoint_partitions(self):
        dataset = make_three_way_dataset()
        self.assert_partition(dataset, list(self.make_splitter().split(dataset)))

    def test_nonconsecutive_duplicate_and_string_dataframe_indices(self):
        for labels in (range(1000, 1125), ['row'] * 125,
                       ['row_' + str(index) for index in range(125)]):
            dataset = make_three_way_dataset()
            dataset.info.index = labels
            with self.subTest(labels=type(labels).__name__), tempfile.TemporaryDirectory() as path:
                folds = list(self.make_splitter(os.path.join(path, 'split')).split(dataset))
                self.assert_partition(dataset, folds)

    def test_metadata_and_original_dataset_are_preserved(self):
        dataset = make_three_way_dataset()
        original = dataset.info.copy(deep=True)
        folds = list(self.make_splitter().split(dataset))
        for fold in folds:
            for subset in fold:
                expected = original[original['clip_id'].isin(
                    subset.info['clip_id'])].reset_index(drop=True)
                pd.testing.assert_frame_equal(subset.info, expected)
        pd.testing.assert_frame_equal(dataset.info, original)
        folds[0][0].info.loc[0, 'trial_id'] = -1
        pd.testing.assert_frame_equal(dataset.info, original)

    def test_cache_reuse_does_not_rewrite_files(self):
        dataset = make_three_way_dataset()
        first = list(self.make_splitter().split(dataset))
        before = {str(file.relative_to(self.split_path)): file.read_bytes()
                  for file in Path(self.split_path).rglob('*') if file.is_file()}
        splitter = self.make_splitter()
        with patch.object(splitter, 'split_info_constructor', side_effect=AssertionError('Unexpected cache rewrite')):
            cached = list(splitter.split(dataset))
        for expected_fold, cached_fold in zip(first, cached):
            for expected, actual in zip(expected_fold, cached_fold):
                pd.testing.assert_frame_equal(expected.info, actual.info)
        self.assertEqual(before, {str(file.relative_to(self.split_path)): file.read_bytes()
                                 for file in Path(self.split_path).rglob('*') if file.is_file()})

    def test_dataframe_index_labels_do_not_invalidate_cache(self):
        dataset = make_three_way_dataset()
        expected = list(self.make_splitter().split(dataset))
        dataset.info.index = ['different'] * len(dataset)
        cached = list(self.make_splitter().split(dataset))
        for first, second in zip(expected, cached):
            for a, b in zip(first, second):
                pd.testing.assert_frame_equal(a.info, b.info)

    def test_equal_sample_ids_from_new_string_objects_reuse_cache(self):
        dataset = make_three_way_dataset()
        expected = list(self.make_splitter().split(dataset))
        dataset.info['_record_id'] = [(value + ' ').rstrip()
                                      for value in dataset.info['_record_id']]
        cached = list(self.make_splitter().split(dataset))
        for a, b in zip(expected, cached):
            for first, second in zip(a, b):
                pd.testing.assert_frame_equal(first.info, second.info)

    def test_changed_identifier_types_require_a_new_cache(self):
        dataset = make_three_way_dataset()
        column = self.metadata_column
        dataset.info[column] = dataset.info[column].astype(object)
        list(self.make_splitter().split(dataset))
        dataset.info[column] = [value.encode() if isinstance(value, str) else str(value)
                                for value in dataset.info[column]]
        with self.assertRaisesRegex(ValueError, 'dataset order'):
            next(self.make_splitter().split(dataset))

    def test_empty_existing_directory_is_initialized(self):
        os.makedirs(self.split_path)
        dataset = make_three_way_dataset()
        self.assert_partition(dataset, list(self.make_splitter().split(dataset)))

    def test_partial_cache_is_rejected_before_any_fold(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        next(Path(self.split_path).rglob('val.csv')).unlink()
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self.make_splitter().split(dataset))

    def test_corrupted_index_file_is_rejected(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        next(Path(self.split_path).rglob('train.csv')).write_text('index\n-1\n')
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self.make_splitter().split(dataset))

    def test_corrupt_manifest_is_rejected(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        Path(self.split_path, '_split_manifest.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self.make_splitter().split(dataset))

    def test_reordered_samples_require_a_new_cache(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        dataset.info = dataset.info.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'dataset order'):
            next(self.make_splitter().split(dataset))

    def test_failed_creation_cleans_up_and_can_be_retried(self):
        splitter = self.make_splitter()
        dataset = make_three_way_dataset()
        def failed_writer(info, split_path):
            Path(split_path, 'train.csv').write_text('index\n0\n')
            raise RuntimeError('Simulated failed creation')
        with patch.object(splitter, 'split_info_constructor', side_effect=failed_writer):
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                list(splitter.split(dataset))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.assert_partition(dataset, list(splitter.split(dataset)))

    def test_parallel_threads_wait_for_complete_cache(self):
        first, second = self.make_splitter(), self.make_splitter()
        dataset = make_three_way_dataset()
        started = threading.Event()
        release = threading.Event()
        writer = first.split_info_constructor
        def delayed_writer(info, split_path):
            directory = Path(split_path, 'fold_0') if self.fold_count > 1 else Path(split_path)
            directory.mkdir(exist_ok=True)
            Path(directory, 'train.csv').write_text('index\n0\n')
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError('Test writer was not released')
            writer(info, split_path)
        with patch.object(first, 'split_info_constructor', side_effect=delayed_writer):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                created = pool.submit(lambda: list(first.split(dataset)))
                try:
                    self.assertTrue(started.wait(timeout=10))
                    cached = pool.submit(lambda: list(second.split(dataset)))
                    with self.assertRaises(concurrent.futures.TimeoutError):
                        cached.result(timeout=0.1)
                finally:
                    release.set()
                self.assert_partition(dataset, created.result(timeout=10))
                self.assert_partition(dataset, cached.result(timeout=10))

    def test_parallel_spawn_processes_reuse_the_same_cache(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        processes = [context.Process(target=split_worker,
                     args=(self.splitter_class.__name__, self.parameters,
                           self.split_path, queue)) for _ in range(2)]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for status, value in results:
                self.assertEqual(status, 'ok', value)
            self.assertEqual(results[0], results[1])
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            queue.close()
            queue.join_thread()

    def test_stale_creation_lock_times_out_without_deleting_it(self):
        lock = Path(self.split_path + '.lock')
        lock.write_text('Test-only lock')
        with patch('torcheeg.model_selection._three_way._CACHE_TIMEOUT', 0.0):
            with self.assertRaisesRegex(RuntimeError, 'Timeout'):
                list(self.make_splitter().split(make_three_way_dataset()))
        self.assertTrue(lock.exists())

    def test_unrecognized_old_cache_is_preserved_and_rejected(self):
        os.makedirs(self.split_path)
        file = Path(self.split_path, 'train.csv')
        file.write_text('index\n0\n')
        before = file.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Incomplete split cache'):
            list(self.make_splitter().split(make_three_way_dataset()))
        self.assertEqual(file.read_bytes(), before)

    def test_missing_or_null_group_metadata_is_rejected(self):
        for null_values in (False, True):
            dataset = make_three_way_dataset()
            if null_values:
                dataset.info[self.metadata_column] = None
            else:
                dataset.info = dataset.info.drop(columns=self.metadata_column)
            with self.subTest(null_values=null_values), tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(ValueError, self.metadata_column):
                    list(self.make_splitter(os.path.join(path, 'split')).split(dataset))
