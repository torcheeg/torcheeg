import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from torcheeg.model_selection import TrainTestSplitPerSubjectGroupbyTrial

from .test_utils import _make_dataset


def _interrupted_split_writer(split_path, ready):
    splitter = TrainTestSplitPerSubjectGroupbyTrial(split_path=split_path)

    def write_partial(info, path):
        Path(path, 'train_subject_0.csv').write_text('index\n0\n')
        import time
        ready.send(True)
        time.sleep(30)

    splitter.build = write_partial
    list(splitter.split(_make_dataset()))


class TestTrainTestSplitPerSubjectGroupbyTrial(unittest.TestCase):

    def test_retry_after_writer_process_is_terminated(self):
        context = mp.get_context('spawn')
        ready, child_ready = context.Pipe(duplex=False)
        process = context.Process(target=_interrupted_split_writer,
                                  args=(self._split_path, child_ready))
        try:
            process.start()
            self.assertTrue(ready.poll(20))
            self.assertTrue(ready.recv())
            process.terminate()
            process.join(timeout=10)
            self.assertFalse(process.is_alive())
            self.assertFalse(Path(self._split_path).exists())
            # The abandoned private staging directory cannot be read as a cache.
            with patch('torcheeg.model_selection.utils._CACHE_TIMEOUT', 1.0):
                self._assert_folds(self._dataset,
                                   list(self._make_splitter().split(self._dataset)))
        finally:
            ready.close()
            child_ready.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')
        self._dataset = _make_dataset()

    def _make_splitter(self, **overrides):
        parameters = dict(train_size=0.6, split_path=self._split_path)
        parameters.update(overrides)
        return TrainTestSplitPerSubjectGroupbyTrial(**parameters)

    def _assert_folds(self, dataset, folds):
        self.assertEqual(len(folds), 5)
        all_clips = set()
        for (train, test), subject in zip(folds, pd.unique(dataset.info['subject_id'])):
            train_clips, test_clips = set(train.info['clip_id']), set(
                test.info['clip_id'])
            self.assertFalse(train_clips & test_clips)
            self.assertEqual(
                train_clips | test_clips,
                set(dataset.info.loc[dataset.info['subject_id'] == subject, 'clip_id']))
            all_clips.update(train_clips | test_clips)
            for subset, count in ((train, 3), (test, 2)):
                self.assertEqual(set(subset.info['subject_id']), {subject})
                self.assertEqual(
                    subset.info.groupby('trial_id').size().tolist(), [count] * 5)
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))
        self.assertEqual(all_clips, set(dataset.info['clip_id']))

    def test_samples_partition_each_subject_and_trial(self):
        self._assert_folds(self._dataset,
                           list(self._make_splitter().split(self._dataset)))

    def test_without_shuffle_preserves_trial_sample_order(self):
        train, test = next(self._make_splitter().split(self._dataset))
        self.assertEqual(
            train.info['clip_id'].tolist(),
            [str(trial * 5 + clip) for trial in range(5) for clip in range(3)])
        self.assertEqual(
            test.info['clip_id'].tolist(),
            [str(trial * 5 + clip) for trial in range(5) for clip in (3, 4)])

    def test_dataframe_labels_are_not_used_as_positions(self):
        for labels in (range(1000, 1125), ['duplicate'] * 125,
                       ['row_' + str(i) for i in range(125)]):
            with self.subTest(labels=type(
                    labels).__name__), tempfile.TemporaryDirectory() as path:
                self._dataset.info.index = labels
                self._assert_folds(
                    self._dataset,
                    list(
                        self._make_splitter(
                            split_path=os.path.join(path, 'split')).split(
                                self._dataset)))

    def test_original_metadata_types_and_values_are_preserved(self):
        original = self._dataset.info.copy(deep=True)
        folds = list(self._make_splitter().split(self._dataset))
        for fold in folds:
            for subset in fold:
                expected = original[original['clip_id'].isin(
                    subset.info['clip_id'])].reset_index(drop=True)
                pd.testing.assert_frame_equal(subset.info, expected)
        folds[0][0].info.loc[0, 'trial_id'] = -1
        pd.testing.assert_frame_equal(self._dataset.info, original)

    def test_shuffle_is_reproducible_and_does_not_change_global_rng(self):
        np.random.seed(71)
        before = np.random.get_state()
        first = list(
            self._make_splitter(shuffle=True, random_state=3).split(self._dataset))
        after = np.random.get_state()
        for expected, actual in zip(before, after):
            np.testing.assert_equal(expected, actual)
        with tempfile.TemporaryDirectory() as path:
            second = list(
                self._make_splitter(shuffle=True,
                                    random_state=3,
                                    split_path=os.path.join(path, 'split')).split(
                                        self._dataset))
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                pd.testing.assert_frame_equal(x.info, y.info)
        self._assert_folds(self._dataset, first)
        patterns = [
            tuple(
                int(value) % 5
                for value in first[0][0].info.loc[first[0][0].info.trial_id == trial,
                                                  'clip_id']) for trial in range(5)
        ]
        self.assertGreater(len(set(patterns)), 1)

    def test_subject_filter_accepts_metadata_id_and_numeric_string(self):
        splitter = self._make_splitter()
        for subject in (2, '2'):
            folds = list(splitter.split(self._dataset, subject=subject))
            self.assertEqual(len(folds), 1)
            self.assertEqual(set(folds[0][0].info.subject_id), {2})
        self.assertEqual(splitter.subjects, ['0', '1', '2', '3', '4'])

    def test_unknown_subject_does_not_create_cache(self):
        with self.assertRaises(ValueError):
            list(self._make_splitter().split(self._dataset, subject=99))
        self.assertFalse(os.path.exists(self._split_path))

    def test_encoded_string_subject_and_trial_ids(self):
        self._dataset.info['subject_id'] = self._dataset.info.subject_id.map(
            lambda i: ['../01', 'a/b', 'a%b', '空 格', '004'][i])
        self._dataset.info['trial_id'] = self._dataset.info.trial_id.map(
            lambda i: 'trial_' + str(i))
        splitter = self._make_splitter()
        self._assert_folds(self._dataset, list(splitter.split(self._dataset)))
        self.assertEqual(splitter.subjects, ['../01', 'a/b', 'a%b', '空 格', '004'])
        self.assertEqual(len(list(splitter.split(self._dataset, subject='a/b'))), 1)
        self.assertFalse(Path(self._directory.name, '01.csv').exists())

    def test_ambiguous_string_representation_is_rejected(self):
        self._dataset.info['subject_id'] = self._dataset.info.subject_id.astype(object)
        self._dataset.info.loc[self._dataset.info.subject_id == 1, 'subject_id'] = '0'
        with self.assertRaisesRegex(ValueError, 'distinct'):
            list(self._make_splitter().split(self._dataset))

    def test_invalid_constructor_parameters(self):
        cases = [{
            'train_size': value
        } for value in (0, 1, -1, float('nan'), True, '0.8')]
        cases += [{'random_state': value} for value in (-1, 2**32, True, 1.5)]
        cases += [{'shuffle': 'yes'}]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self._make_splitter(**parameters)

    def test_empty_missing_and_null_group_columns(self):
        frames = [
            self._dataset.info.iloc[:0],
            self._dataset.info.drop(columns=['subject_id']),
            self._dataset.info.drop(columns=['trial_id'])
        ]
        for column in ('subject_id', 'trial_id'):
            frame = self._dataset.info.copy()
            frame.loc[0, column] = np.nan
            frames.append(frame)
        for frame in frames:
            self._dataset.info = frame
            with self.assertRaises(ValueError):
                list(self._make_splitter().split(self._dataset))

    def test_every_trial_requires_nonempty_training_and_test_samples(self):
        for ratio in (0.1, 0.99):
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as path:
                dataset = _make_dataset()
                if ratio == 0.99:
                    dataset.info = dataset.info.drop(index=[1, 2, 3, 4]).reset_index(
                        drop=True)
                with self.assertRaisesRegex(ValueError, 'at least one'):
                    list(
                        self._make_splitter(train_size=ratio,
                                            split_path=os.path.join(
                                                path, 'split')).split(dataset))
                self.assertFalse(os.path.exists(os.path.join(path, 'split')))

    def test_corrupted_csv_or_subject_list_is_rejected(self):
        for name in ('train_subject_0.csv', '_subjects.json', '_split_manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as path:
                splitter = self._make_splitter(split_path=os.path.join(path, 'split'))
                list(splitter.split(self._dataset))
                Path(path, 'split', name).write_text('corrupt')
                with self.assertRaises(ValueError):
                    list(splitter.split(self._dataset))

    def test_repr_documents_configuration(self):
        text = repr(self._make_splitter(shuffle=True, random_state=42))
        for part in ('TrainTestSplitPerSubjectGroupbyTrial', 'train_size=0.6',
                     'shuffle=True', 'random_state=42', self._split_path):
            self.assertIn(part, text)


if __name__ == '__main__':
    unittest.main()
