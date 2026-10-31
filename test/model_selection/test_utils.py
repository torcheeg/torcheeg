import concurrent.futures
import multiprocessing as mp
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from torcheeg import model_selection
from torcheeg.datasets import BaseDataset
from torcheeg.io import EEGSignalIO


def _make_dataset():
    dataset = BaseDataset.__new__(BaseDataset)
    rows = []
    io = EEGSignalIO('unused', io_mode='memory')
    for subject in range(5):
        for trial in range(5):
            for clip in range(5):
                index = len(rows)
                io.write_eeg(np.array([index, trial, clip]), key=str(index))
                rows.append({
                    'subject_id': subject,
                    'trial_id': trial,
                    'clip_id': str(index),
                    '_record_id': '_record_0',
                    'split': ('train', 'val', 'test')[index % 3]
                })
    dataset.info = pd.DataFrame(rows)
    dataset.info['category'] = pd.Categorical(['rest', 'task', 'task', 'rest', 'rest'] *
                                              25)
    dataset.info['timestamp'] = pd.date_range('2020-01-01', periods=len(rows))
    dataset.eeg_io_router = {'_record_0': io}
    dataset.eeg_signal_paths = {}
    return dataset


def _split_worker(class_name, parameters, split_path, queue):
    try:
        splitter = getattr(model_selection, class_name)(split_path=split_path,
                                                        **parameters)
        folds = list(splitter.split(_make_dataset()))
        queue.put(('ok', [[list(subset.info['clip_id']) for subset in fold]
                          for fold in folds]))
    except Exception as error:
        queue.put(('error', repr(error)))


class TestModelSelectionUtils(unittest.TestCase):
    _splitter_class = model_selection.TrainValTestSplitCrossSubject
    _parameters = {
        'train_subjects': [0, 1, 2],
        'val_subjects': [3],
        'test_subjects': [4]
    }
    _fold_count = 1
    _group_column = 'subject_id'
    _metadata_column = 'subject_id'

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def _make_splitter(self, split_path=None, **overrides):
        parameters = dict(self._parameters)
        parameters.update(overrides)
        return self._splitter_class(
            split_path=self._split_path if split_path is None else split_path,
            **parameters)

    def _assert_partition(self, dataset, folds):
        self.assertEqual(len(folds), self._fold_count)
        for fold in folds:
            self.assertEqual(len(fold), 3)
            clips = [set(subset.info['clip_id']) for subset in fold]
            self.assertEqual(set.union(*clips), set(dataset.info['clip_id']))
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(clips[first] & clips[second])
                if self._group_column is not None:
                    self.assertFalse(
                        set(fold[first].info[self._group_column])
                        & set(fold[second].info[self._group_column]))
            for subset in fold:
                self.assertGreater(len(subset), 0)
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))

    def test_samples_and_groups_form_complete_disjoint_partitions(self):
        dataset = _make_dataset()
        self._assert_partition(dataset, list(
            self._make_splitter().split(dataset)))

    def test_nonconsecutive_duplicate_and_string_dataframe_indices(self):
        for labels in (range(1000, 1125), ['row'] * 125,
                       ['row_' + str(index) for index in range(125)]):
            dataset = _make_dataset()
            dataset.info.index = labels
            with self.subTest(labels=type(
                    labels).__name__), tempfile.TemporaryDirectory() as path:
                folds = list(
                    self._make_splitter(os.path.join(path, 'split')).split(dataset))
                self._assert_partition(dataset, folds)

    def test_metadata_and_original_dataset_are_preserved(self):
        dataset = _make_dataset()
        original = dataset.info.copy(deep=True)
        folds = list(self._make_splitter().split(dataset))
        for fold in folds:
            for subset in fold:
                expected = original[original['clip_id'].isin(
                    subset.info['clip_id'])].reset_index(drop=True)
                pd.testing.assert_frame_equal(subset.info, expected)
        pd.testing.assert_frame_equal(dataset.info, original)
        folds[0][0].info.loc[0, 'trial_id'] = -1
        pd.testing.assert_frame_equal(dataset.info, original)

    def test_cache_reuse_does_not_rewrite_files(self):
        dataset = _make_dataset()
        first = list(self._make_splitter().split(dataset))
        before = {
            str(file.relative_to(self._split_path)): file.read_bytes()
            for file in Path(self._split_path).rglob('*') if file.is_file()
        }
        splitter = self._make_splitter()
        with patch.object(splitter,
                          'build',
                          side_effect=AssertionError('Unexpected cache rewrite')):
            cached = list(splitter.split(dataset))
        for expected_fold, cached_fold in zip(first, cached):
            for expected, actual in zip(expected_fold, cached_fold):
                pd.testing.assert_frame_equal(expected.info, actual.info)
        self.assertEqual(
            before, {
                str(file.relative_to(self._split_path)): file.read_bytes()
                for file in Path(self._split_path).rglob('*') if file.is_file()
            })

    def test_dataframe_index_labels_do_not_invalidate_cache(self):
        dataset = _make_dataset()
        expected = list(self._make_splitter().split(dataset))
        dataset.info.index = ['different'] * len(dataset)
        cached = list(self._make_splitter().split(dataset))
        for first, second in zip(expected, cached):
            for a, b in zip(first, second):
                pd.testing.assert_frame_equal(a.info, b.info)

    def test_equal_sample_ids_from_new_string_objects_reuse_cache(self):
        dataset = _make_dataset()
        expected = list(self._make_splitter().split(dataset))
        dataset.info['_record_id'] = [(value + ' ').rstrip()
                                      for value in dataset.info['_record_id']]
        cached = list(self._make_splitter().split(dataset))
        for a, b in zip(expected, cached):
            for first, second in zip(a, b):
                pd.testing.assert_frame_equal(first.info, second.info)

    def test_changed_identifier_types_require_a_new_cache(self):
        dataset = _make_dataset()
        column = self._metadata_column
        dataset.info[column] = dataset.info[column].astype(object)
        list(self._make_splitter().split(dataset))
        dataset.info[column] = [
            value.encode() if isinstance(value, str) else str(value)
            for value in dataset.info[column]
        ]
        with self.assertRaisesRegex(ValueError, 'changed'):
            next(self._make_splitter().split(dataset))

    def test_empty_existing_directory_is_initialized(self):
        os.makedirs(self._split_path)
        dataset = _make_dataset()
        self._assert_partition(dataset, list(
            self._make_splitter().split(dataset)))

    def test_partial_cache_is_rejected_before_any_fold(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        next(Path(self._split_path).rglob('val.csv')).unlink()
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self._make_splitter().split(dataset))

    def test_corrupted_index_file_is_rejected(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        next(Path(self._split_path).rglob('train.csv')).write_text('index\n-1\n')
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self._make_splitter().split(dataset))

    def test_corrupt_manifest_is_rejected(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        Path(self._split_path, 'manifest.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'cache'):
            next(self._make_splitter().split(dataset))

    def test_reordered_samples_require_a_new_cache(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        dataset.info = dataset.info.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'changed'):
            next(self._make_splitter().split(dataset))

    def test_failed_creation_cleans_up_and_can_be_retried(self):
        splitter = self._make_splitter()
        dataset = _make_dataset()

        def failed_writer(info, split_path):
            Path(split_path, 'train.csv').write_text('index\n0\n')
            raise RuntimeError('Simulated failed creation')

        with patch.object(splitter, 'build', side_effect=failed_writer):
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                list(splitter.split(dataset))
        self.assertFalse(Path(self._split_path).exists())
        self.assertFalse(
            list(Path(self._directory.name).glob('.torcheeg-split-*')))
        self._assert_partition(dataset, list(splitter.split(dataset)))

    def test_parallel_threads_wait_for_complete_cache(self):
        first, second = self._make_splitter(), self._make_splitter()
        dataset = _make_dataset()
        started = threading.Event()
        release = threading.Event()
        writer = first.build

        def delayed_writer(info, split_path):
            directory = Path(split_path,
                             'fold_0') if self._fold_count > 1 else Path(split_path)
            directory.mkdir(exist_ok=True)
            Path(directory, 'train.csv').write_text('index\n0\n')
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError('Test writer was not released')
            writer(info, split_path)

        with patch.object(first, 'build', side_effect=delayed_writer):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                created = pool.submit(lambda: list(first.split(dataset)))
                try:
                    self.assertTrue(started.wait(timeout=10))
                    cached = pool.submit(lambda: list(second.split(dataset)))
                    with self.assertRaises(concurrent.futures.TimeoutError):
                        cached.result(timeout=0.1)
                finally:
                    release.set()
                self._assert_partition(dataset, created.result(timeout=10))
                self._assert_partition(dataset, cached.result(timeout=10))

    def test_parallel_spawn_processes_reuse_the_same_cache(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        processes = [
            context.Process(target=_split_worker,
                            args=(self._splitter_class.__name__, self._parameters,
                                  self._split_path, queue)) for _ in range(2)
        ]
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

    def test_unrecognized_old_cache_is_preserved_and_rejected(self):
        os.makedirs(self._split_path)
        file = Path(self._split_path, 'train.csv')
        file.write_text('index\n0\n')
        before = file.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Incomplete cache'):
            list(self._make_splitter().split(_make_dataset()))
        self.assertEqual(file.read_bytes(), before)

    def test_missing_or_null_group_metadata_is_rejected(self):
        for null_values in (False, True):
            dataset = _make_dataset()
            if null_values:
                dataset.info[self._metadata_column] = None
            else:
                dataset.info = dataset.info.drop(columns=self._metadata_column)
            with self.subTest(
                    null_values=null_values), tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(ValueError, self._metadata_column):
                    list(
                        self._make_splitter(os.path.join(path, 'split')).split(dataset))

    def test_invalid_positions_are_rejected(self):
        from torcheeg.model_selection.utils import validate_indices
        for indices in ([], [0, 0], [-1], [125], [0.5], [[0, 1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                validate_indices(indices, 125)

    def test_subject_string_ids_are_not_coerced_by_csv(self):
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map(
            lambda value: f'{value:03d}')
        with tempfile.TemporaryDirectory() as directory:
            splitter = model_selection.KFoldCrossSubject(n_splits=5,
                                                         split_path=os.path.join(
                                                             directory, 'split'))
            for train, test in splitter.split(dataset):
                self.assertEqual(
                    set(train.info.subject_id) | set(test.info.subject_id),
                    {'000', '001', '002', '003', '004'})

    def test_stratification_parameters_are_checked_for_classes_and_functions(self):
        dataset = _make_dataset()
        for class_based in (True, False):
            with self.subTest(
                    class_based=class_based), tempfile.TemporaryDirectory() as path:
                split_path = os.path.join(path, "split")
                transform = lambda y: {"y": y["subject_id"] % 2}
                changed = lambda y: {"y": (y["subject_id"] + 1) % 2}
                if class_based:
                    splitter = model_selection.KFoldCrossSubject(
                        n_splits=2,
                        shuffle=True,
                        random_state=42,
                        label_transform=transform,
                        split_path=split_path)
                    folds = list(splitter.split(dataset))
                    repeated = list(splitter.split(dataset))
                    for first, second in zip(folds, repeated):
                        for a, b in zip(first, second):
                            pd.testing.assert_frame_equal(a.info, b.info)
                    splitter.label_transform = changed
                    with self.assertRaisesRegex(ValueError, "parameters"):
                        list(splitter.split(dataset))
                else:
                    arguments = dict(test_size=0.4,
                                     shuffle=True,
                                     random_state=42,
                                     split_path=split_path)
                    first = model_selection.train_test_split_cross_subject(
                        dataset, label_transform=transform, **arguments)
                    second = model_selection.train_test_split_cross_subject(
                        dataset, label_transform=transform, **arguments)
                    for a, b in zip(first, second):
                        pd.testing.assert_frame_equal(a.info, b.info)
                    with self.assertRaisesRegex(ValueError, "parameters"):
                        model_selection.train_test_split_cross_subject(
                            dataset, label_transform=changed, **arguments)

    def test_reordered_samples_without_clip_or_record_ids_reject_cache(self):
        dataset = _make_dataset()
        dataset.info = dataset.info.drop(columns=['clip_id', '_record_id'])
        splitter = model_selection.KFoldGroupbyTrial(n_splits=5,
                                                     split_path=self._split_path)
        list(splitter.split(dataset))
        dataset.info.iloc[[0, 1]] = dataset.info.iloc[[1, 0]].to_numpy()
        with self.assertRaisesRegex(ValueError, 'changed'):
            list(splitter.split(dataset))
