import os
import tempfile
import unittest

import pandas as pd

from torcheeg.model_selection import LeaveOneSubjectOutIndex

from ._index_dataset import make_index_dataset


class TestLeaveOneSubjectOutIndex(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.split_path = os.path.join(directory.name, 'split')

    def assert_subject_partition(self, dataset, folds):
        all_clips = set(dataset.info['clip_id'])
        test_subjects = []
        for train, test in folds:
            subjects = set(test.info['subject_id'])
            self.assertEqual(len(subjects), 1)
            self.assertFalse(set(train.info['subject_id']) & subjects)
            self.assertEqual(set(train.info['clip_id']) | set(test.info['clip_id']), all_clips)
            self.assertFalse(set(train.info['clip_id']) & set(test.info['clip_id']))
            test_subjects.extend(subjects)
            for subset in (train, test):
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))
        self.assertCountEqual(test_subjects, list(dataset.info['subject_id'].unique()))

    def test_leave_one_subject_out_index(self):
        dataset = make_index_dataset()
        folds = list(LeaveOneSubjectOutIndex(split_path=self.split_path).split(dataset))
        self.assertEqual(len(folds), 5)
        for train, test in folds:
            self.assertEqual(len(train), 100)
            self.assertEqual(len(test), 25)
        self.assert_subject_partition(dataset, folds)

    def test_positions_do_not_depend_on_dataframe_index(self):
        for index in (range(1000, 1125), ['row'] * 125,
                      ['row_' + str(i) for i in range(125)]):
            dataset = make_index_dataset()
            dataset.info.index = index
            with self.subTest(index_type=type(index).__name__), tempfile.TemporaryDirectory() as directory:
                cv = LeaveOneSubjectOutIndex(split_path=os.path.join(directory, 'split'))
                self.assert_subject_partition(dataset, list(cv.split(dataset)))

    def test_subject_selection_and_metadata_types(self):
        dataset = make_index_dataset(subjects=['01', '02', '10'])
        original_info = dataset.info.copy(deep=True)
        cv = LeaveOneSubjectOutIndex(split_path=self.split_path)
        folds = list(cv.split(dataset, subject='01'))
        self.assertEqual(len(folds), 1)
        train, test = folds[0]
        self.assertEqual(set(test.info['subject_id']), {'01'})
        for subset in (train, test):
            expected = original_info[original_info['clip_id'].isin(
                subset.info['clip_id'])].reset_index(drop=True)
            pd.testing.assert_frame_equal(subset.info, expected)
        pd.testing.assert_frame_equal(dataset.info, original_info)

    def test_existing_cache_is_reused(self):
        dataset = make_index_dataset()
        first = list(LeaveOneSubjectOutIndex(split_path=self.split_path).split(dataset))
        restored = list(LeaveOneSubjectOutIndex(split_path=self.split_path).split(dataset))
        for expected, cached in zip(first, restored):
            for a, b in zip(expected, cached):
                pd.testing.assert_frame_equal(a.info, b.info)

    def test_unrelated_files_do_not_become_subjects(self):
        cv = LeaveOneSubjectOutIndex(split_path=self.split_path)
        list(cv.split(make_index_dataset()))
        with open(os.path.join(self.split_path, 'README.txt'), 'w') as file:
            file.write('Unrelated file.')
        self.assertEqual(len(list(cv.split(make_index_dataset()))), 5)

    def test_incomplete_cache_is_rejected_before_yielding(self):
        cv = LeaveOneSubjectOutIndex(split_path=self.split_path)
        dataset = make_index_dataset()
        list(cv.split(dataset))
        os.remove(os.path.join(self.split_path, 'test_subject_4.csv'))
        with self.assertRaisesRegex(ValueError, 'Incomplete split cache'):
            next(cv.split(dataset))

    def test_empty_existing_directory_is_initialized(self):
        os.makedirs(self.split_path)
        cv = LeaveOneSubjectOutIndex(split_path=self.split_path)
        self.assertEqual(len(list(cv.split(make_index_dataset()))), 5)

    def test_subject_identifiers_are_safe_in_cache_paths(self):
        dataset = make_index_dataset(subjects=['sub/01', 'sub%02', 'sub_03'])
        cv = LeaveOneSubjectOutIndex(split_path=self.split_path)
        self.assert_subject_partition(dataset, list(cv.split(dataset)))
        train, test = next(cv.split(dataset, subject='sub/01'))
        self.assertEqual(set(test.info['subject_id']), {'sub/01'})
        self.assertEqual(len(os.listdir(self.split_path)), 6)

    def test_unknown_subject_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'subject'):
            list(LeaveOneSubjectOutIndex(split_path=self.split_path).split(
                make_index_dataset(), subject='unknown'))

    def test_at_least_two_subjects_are_required(self):
        with self.assertRaisesRegex(ValueError, 'subject'):
            list(LeaveOneSubjectOutIndex(split_path=self.split_path).split(
                make_index_dataset(subjects=['only'])))

    def test_missing_subject_metadata_is_rejected(self):
        for info in (pd.DataFrame({'clip_id': ['0', '1']}),
                     pd.DataFrame({'subject_id': [0, None]})):
            dataset = make_index_dataset()
            dataset.info = info
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'subject_id'):
                    list(LeaveOneSubjectOutIndex(split_path=os.path.join(
                        directory, 'split')).split(dataset))


if __name__ == '__main__':
    unittest.main()
