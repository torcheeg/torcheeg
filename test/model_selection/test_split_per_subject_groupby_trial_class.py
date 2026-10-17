import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from torcheeg.model_selection import TrainTestSplitPerSubjectGroupbyTrial

from ._three_way_cases import make_three_way_dataset, split_worker


class TestTrainTestSplitPerSubjectGroupbyTrial(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.split_path = os.path.join(self.directory.name, 'split')
        self.dataset = make_three_way_dataset()

    def make_splitter(self, **overrides):
        parameters = dict(train_size=0.6, split_path=self.split_path)
        parameters.update(overrides)
        return TrainTestSplitPerSubjectGroupbyTrial(**parameters)

    def assert_folds(self, dataset, folds):
        self.assertEqual(len(folds), 5)
        all_clips = set()
        for (train, test), subject in zip(folds, pd.unique(dataset.info['subject_id'])):
            train_clips, test_clips = set(train.info['clip_id']), set(test.info['clip_id'])
            self.assertFalse(train_clips & test_clips)
            self.assertEqual(train_clips | test_clips, set(dataset.info.loc[
                dataset.info['subject_id'] == subject, 'clip_id']))
            all_clips.update(train_clips | test_clips)
            for subset, count in ((train, 3), (test, 2)):
                self.assertEqual(set(subset.info['subject_id']), {subject})
                self.assertEqual(subset.info.groupby('trial_id').size().tolist(), [count] * 5)
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))
        self.assertEqual(all_clips, set(dataset.info['clip_id']))

    def test_samples_partition_each_subject_and_trial(self):
        self.assert_folds(self.dataset, list(self.make_splitter().split(self.dataset)))

    def test_without_shuffle_preserves_trial_sample_order(self):
        train, test = next(self.make_splitter().split(self.dataset))
        self.assertEqual(train.info['clip_id'].tolist(),
                         [str(trial * 5 + clip) for trial in range(5) for clip in range(3)])
        self.assertEqual(test.info['clip_id'].tolist(),
                         [str(trial * 5 + clip) for trial in range(5) for clip in (3, 4)])

    def test_dataframe_labels_are_not_used_as_positions(self):
        for labels in (range(1000, 1125), ['duplicate'] * 125, ['row_' + str(i) for i in range(125)]):
            with self.subTest(labels=type(labels).__name__), tempfile.TemporaryDirectory() as path:
                self.dataset.info.index = labels
                self.assert_folds(self.dataset, list(self.make_splitter(
                    split_path=os.path.join(path, 'split')).split(self.dataset)))

    def test_original_metadata_types_and_values_are_preserved(self):
        original = self.dataset.info.copy(deep=True)
        folds = list(self.make_splitter().split(self.dataset))
        for fold in folds:
            for subset in fold:
                expected = original[original['clip_id'].isin(subset.info['clip_id'])].reset_index(drop=True)
                pd.testing.assert_frame_equal(subset.info, expected)
        folds[0][0].info.loc[0, 'trial_id'] = -1
        pd.testing.assert_frame_equal(self.dataset.info, original)

    def test_shuffle_is_reproducible_and_does_not_change_global_rng(self):
        np.random.seed(71)
        before = np.random.get_state()
        first = list(self.make_splitter(shuffle=True, random_state=3).split(self.dataset))
        after = np.random.get_state()
        for expected, actual in zip(before, after):
            np.testing.assert_equal(expected, actual)
        with tempfile.TemporaryDirectory() as path:
            second = list(self.make_splitter(shuffle=True, random_state=3,
                split_path=os.path.join(path, 'split')).split(self.dataset))
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                pd.testing.assert_frame_equal(x.info, y.info)
        self.assert_folds(self.dataset, first)
        patterns = [tuple(int(value) % 5 for value in first[0][0].info.loc[
            first[0][0].info.trial_id == trial, 'clip_id']) for trial in range(5)]
        self.assertGreater(len(set(patterns)), 1)

    def test_subject_filter_accepts_metadata_id_and_numeric_string(self):
        splitter = self.make_splitter()
        for subject in (2, '2'):
            folds = list(splitter.split(self.dataset, subject=subject))
            self.assertEqual(len(folds), 1)
            self.assertEqual(set(folds[0][0].info.subject_id), {2})
        self.assertEqual(splitter.subjects, ['0', '1', '2', '3', '4'])

    def test_unknown_subject_does_not_create_cache(self):
        with self.assertRaises(ValueError):
            list(self.make_splitter().split(self.dataset, subject=99))
        self.assertFalse(os.path.exists(self.split_path))

    def test_encoded_string_subject_and_trial_ids(self):
        self.dataset.info['subject_id'] = self.dataset.info.subject_id.map(
            lambda i: ['../01', 'a/b', 'a%b', '空 格', '004'][i])
        self.dataset.info['trial_id'] = self.dataset.info.trial_id.map(lambda i: 'trial_' + str(i))
        splitter = self.make_splitter()
        self.assert_folds(self.dataset, list(splitter.split(self.dataset)))
        self.assertEqual(splitter.subjects, ['../01', 'a/b', 'a%b', '空 格', '004'])
        self.assertEqual(len(list(splitter.split(self.dataset, subject='a/b'))), 1)
        self.assertFalse(Path(self.directory.name, '01.csv').exists())

    def test_ambiguous_string_representation_is_rejected(self):
        self.dataset.info['subject_id'] = self.dataset.info.subject_id.astype(object)
        self.dataset.info.loc[self.dataset.info.subject_id == 1, 'subject_id'] = '0'
        with self.assertRaisesRegex(ValueError, 'distinct'):
            list(self.make_splitter().split(self.dataset))

    def test_invalid_constructor_parameters(self):
        cases = [{'train_size': value} for value in (0, 1, -1, float('nan'), True, '0.8')]
        cases += [{'random_state': value} for value in (-1, 2 ** 32, True, 1.5)]
        cases += [{'shuffle': 'yes'}]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.make_splitter(**parameters)

    def test_empty_missing_and_null_group_columns(self):
        frames = [self.dataset.info.iloc[:0], self.dataset.info.drop(columns=['subject_id']),
                  self.dataset.info.drop(columns=['trial_id'])]
        for column in ('subject_id', 'trial_id'):
            frame = self.dataset.info.copy()
            frame.loc[0, column] = np.nan
            frames.append(frame)
        for frame in frames:
            self.dataset.info = frame
            with self.assertRaises(ValueError):
                list(self.make_splitter().split(self.dataset))

    def test_every_trial_requires_nonempty_training_and_test_samples(self):
        for ratio in (0.1, 0.99):
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as path:
                dataset = make_three_way_dataset()
                if ratio == 0.99:
                    dataset.info = dataset.info.drop(index=[1, 2, 3, 4]).reset_index(drop=True)
                with self.assertRaisesRegex(ValueError, 'at least one'):
                    list(self.make_splitter(train_size=ratio,
                        split_path=os.path.join(path, 'split')).split(dataset))
                self.assertFalse(os.path.exists(os.path.join(path, 'split')))

    def test_empty_existing_directory_is_initialized(self):
        os.makedirs(self.split_path)
        self.assert_folds(self.dataset, list(self.make_splitter().split(self.dataset)))

    def test_cache_reuse_and_equal_string_objects(self):
        self.dataset.info['trial_id'] = self.dataset.info.trial_id.map(lambda i: 'long_trial_' + str(i))
        splitter = self.make_splitter()
        first = list(splitter.split(self.dataset))
        before = {p.name: p.read_bytes() for p in Path(self.split_path).iterdir()}
        self.dataset.info['trial_id'] = self.dataset.info.trial_id.map(lambda s: s.encode().decode())
        with patch.object(splitter, 'split_info_constructor', side_effect=AssertionError('rewritten')):
            second = list(splitter.split(self.dataset))
        self.assertEqual(before, {p.name: p.read_bytes() for p in Path(self.split_path).iterdir()})
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                pd.testing.assert_frame_equal(x.info, y.info)

    def test_changes_to_trials_subjects_samples_order_or_parameters_reject_cache(self):
        mutations = [lambda f: f.assign(trial_id=f.trial_id + 1),
                     lambda f: f.assign(subject_id=f.subject_id + 1),
                     lambda f: f.assign(clip_id='different'),
                     lambda f: f.iloc[::-1].reset_index(drop=True)]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as path:
                splitter = self.make_splitter(split_path=os.path.join(path, 'split'))
                dataset = make_three_way_dataset()
                list(splitter.split(dataset))
                dataset.info = mutate(dataset.info)
                with self.assertRaises(ValueError):
                    list(splitter.split(dataset))
        list(self.make_splitter().split(self.dataset))
        with self.assertRaises(ValueError):
            list(self.make_splitter(train_size=0.8).split(self.dataset))

    def test_partial_cache_is_rejected_without_overwrite(self):
        os.makedirs(self.split_path)
        file = Path(self.split_path, 'train_subject_0.csv')
        file.write_text('index\n0\n')
        with self.assertRaises(ValueError):
            list(self.make_splitter().split(self.dataset))
        self.assertEqual(file.read_text(), 'index\n0\n')

    def test_corrupted_csv_or_subject_list_is_rejected(self):
        for name in ('train_subject_0.csv', '_subjects.json', '_split_manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as path:
                splitter = self.make_splitter(split_path=os.path.join(path, 'split'))
                list(splitter.split(self.dataset))
                Path(path, 'split', name).write_text('corrupt')
                with self.assertRaises(ValueError):
                    list(splitter.split(self.dataset))

    def test_writer_failure_cleans_staging_and_allows_retry(self):
        splitter = self.make_splitter()
        def fail(info, path):
            Path(path, 'train_subject_0.csv').write_text('index\n0\n')
            raise RuntimeError('intentional failure')
        with patch.object(splitter, 'split_info_constructor', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                list(splitter.split(self.dataset))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.assert_folds(self.dataset, list(splitter.split(self.dataset)))

    def test_independent_processes_share_a_complete_cache(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        workers = [context.Process(target=split_worker, args=(
            'TrainTestSplitPerSubjectGroupbyTrial', {'train_size': 0.6, 'shuffle': True,
            'random_state': 42}, self.split_path, queue)) for _ in range(2)]
        for worker in workers:
            worker.start()
        try:
            results = [queue.get(timeout=90) for _ in workers]
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(results[0][0], 'ok', results)
            self.assertEqual(results[0], results[1])
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()

    def test_repr_documents_configuration(self):
        text = repr(self.make_splitter(shuffle=True, random_state=42))
        for part in ('TrainTestSplitPerSubjectGroupbyTrial', 'train_size=0.6',
                     'shuffle=True', 'random_state=42', self.split_path):
            self.assertIn(part, text)


if __name__ == '__main__':
    unittest.main()
