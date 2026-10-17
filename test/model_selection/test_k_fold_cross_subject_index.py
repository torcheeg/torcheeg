import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from torcheeg.model_selection import KFoldCrossSubjectIndex

from ._index_dataset import make_index_dataset


class TestKFoldCrossSubjectIndex(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.split_path = os.path.join(directory.name, 'split')

    def assert_subject_partition(self, dataset, folds):
        all_clips = set(dataset.info['clip_id'])
        test_clips = []
        for train, test in folds:
            self.assertFalse(set(train.info['subject_id']) &
                             set(test.info['subject_id']))
            self.assertFalse(set(train.info['clip_id']) & set(test.info['clip_id']))
            self.assertEqual(set(train.info['clip_id']) | set(test.info['clip_id']),
                             all_clips)
            test_clips.extend(test.info['clip_id'])
            for subset in (train, test):
                self.assertEqual(list(subset.info.index), list(range(len(subset))))
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))
        self.assertCountEqual(test_clips, list(dataset.info['clip_id']))

    def test_k_fold_cross_subject_index(self):
        dataset = make_index_dataset()
        folds = list(KFoldCrossSubjectIndex(split_path=self.split_path).split(dataset))
        self.assertEqual(len(folds), 5)
        for train, test in folds:
            self.assertEqual(len(train), 100)
            self.assertEqual(len(test), 25)
        self.assert_subject_partition(dataset, folds)

    def test_positions_do_not_depend_on_dataframe_index(self):
        for index in (range(1000, 1125), ['row'] * 125,
                      ['row_' + str(i) for i in range(125)]):
            with self.subTest(index_type=type(index).__name__):
                dataset = make_index_dataset()
                dataset.info.index = index
                with tempfile.TemporaryDirectory() as directory:
                    cv = KFoldCrossSubjectIndex(split_path=os.path.join(directory, 'split'))
                    self.assert_subject_partition(dataset, list(cv.split(dataset)))

    def test_metadata_and_original_dataset_are_preserved(self):
        dataset = make_index_dataset()
        original_info = dataset.info.copy(deep=True)
        folds = list(KFoldCrossSubjectIndex(split_path=self.split_path).split(dataset))
        for train, test in folds:
            for subset in (train, test):
                expected = original_info[original_info['clip_id'].isin(
                    subset.info['clip_id'])].reset_index(drop=True)
                pd.testing.assert_frame_equal(subset.info, expected)
        pd.testing.assert_frame_equal(dataset.info, original_info)
        folds[0][0].info.loc[0, 'trial_id'] = -1
        pd.testing.assert_frame_equal(dataset.info, original_info)

    def test_seeded_splits_are_reproducible_and_cache_is_reused(self):
        dataset = make_index_dataset(subjects=['s5', 's1', 's3', 's0', 's4', 's2'])
        cv = KFoldCrossSubjectIndex(n_splits=3, shuffle=True, random_state=42,
                                   split_path=self.split_path)
        first = list(cv.split(dataset))
        files = {name: Path(self.split_path, name).read_bytes()
                 for name in os.listdir(self.split_path)}
        with tempfile.TemporaryDirectory() as directory:
            fresh = list(KFoldCrossSubjectIndex(n_splits=3, shuffle=True,
                random_state=42, split_path=os.path.join(directory, 'split')).split(dataset))
        cached = list(KFoldCrossSubjectIndex(n_splits=3, shuffle=True,
            random_state=1, split_path=self.split_path).split(dataset))
        for expected, regenerated, restored in zip(first, fresh, cached):
            for a, b, c in zip(expected, regenerated, restored):
                pd.testing.assert_frame_equal(a.info, b.info)
                pd.testing.assert_frame_equal(a.info, c.info)
        self.assertEqual(files, {name: Path(self.split_path, name).read_bytes()
                                for name in os.listdir(self.split_path)})

    def test_unrelated_files_do_not_become_folds(self):
        cv = KFoldCrossSubjectIndex(split_path=self.split_path)
        dataset = make_index_dataset()
        list(cv.split(dataset))
        with open(os.path.join(self.split_path, 'notes_fold_99.csv.bak'), 'w') as file:
            file.write('Unrelated file.')
        self.assertEqual(len(list(cv.split(dataset))), 5)

    def test_incomplete_cache_is_rejected_before_yielding(self):
        cv = KFoldCrossSubjectIndex(split_path=self.split_path)
        dataset = make_index_dataset()
        list(cv.split(dataset))
        os.remove(os.path.join(self.split_path, 'test_fold_4.csv'))
        with self.assertRaisesRegex(ValueError, 'Incomplete split cache'):
            next(cv.split(dataset))

    def test_missing_entire_fold_is_rejected(self):
        cv = KFoldCrossSubjectIndex(split_path=self.split_path)
        dataset = make_index_dataset()
        list(cv.split(dataset))
        for split in ('train', 'test'):
            os.remove(os.path.join(self.split_path, f'{split}_fold_4.csv'))
        with self.assertRaisesRegex(ValueError, 'Incomplete split cache'):
            next(cv.split(dataset))

    def test_empty_existing_directory_is_initialized(self):
        os.makedirs(self.split_path)
        folds = list(KFoldCrossSubjectIndex(split_path=self.split_path).split(make_index_dataset()))
        self.assertEqual(len(folds), 5)

    def test_missing_subject_metadata_is_rejected(self):
        for info in (pd.DataFrame({'clip_id': ['0', '1']}),
                     pd.DataFrame({'subject_id': [0, None]})):
            dataset = make_index_dataset()
            dataset.info = info
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'subject_id'):
                    list(KFoldCrossSubjectIndex(n_splits=2,
                        split_path=os.path.join(directory, 'split')).split(dataset))

    def test_invalid_number_of_folds_is_rejected(self):
        with self.assertRaises(ValueError):
            KFoldCrossSubjectIndex(n_splits=1, split_path=self.split_path)
        with self.assertRaises(ValueError):
            list(KFoldCrossSubjectIndex(n_splits=6,
                split_path=self.split_path).split(make_index_dataset()))

    def test_repr_contains_configuration(self):
        cv = KFoldCrossSubjectIndex(n_splits=3, shuffle=True, random_state=42,
                                   split_path=self.split_path)
        self.assertIn('n_splits=3', repr(cv))
        self.assertIn('random_state=42', repr(cv))


if __name__ == '__main__':
    unittest.main()
