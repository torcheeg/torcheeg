import os
import tempfile
import unittest

import numpy as np

from torcheeg.model_selection import KFoldCrossSubjectTrainValTest

from .test_utils import _make_dataset


class TestKFoldCrossSubjectTrainValTest(unittest.TestCase):
    _splitter_class = KFoldCrossSubjectTrainValTest
    _parameters = {'all_subjects': [0, 1, 2, 3, 4], 'n_folds': 5}
    _fold_count = 5
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

    def test_k_fold_cross_subject_train_val_test(self):
        folds = list(self._make_splitter().split(_make_dataset()))
        self.assertEqual(len(folds), 5)
        validation_subjects, test_subjects = [], []
        for train, val, test in folds:
            self.assertEqual([len(train), len(val), len(test)], [75, 25, 25])
            validation_subjects.extend(val.info['subject_id'].unique())
            test_subjects.extend(test.info['subject_id'].unique())
        self.assertCountEqual(validation_subjects, list(range(5)))
        self.assertCountEqual(test_subjects, list(range(5)))

    def test_string_subject_ids_keep_their_values_and_types(self):
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map(
            lambda value: f'sub_{value:02d}')
        splitter = self._make_splitter(
            all_subjects=[f'sub_{value:02d}' for value in range(5)])
        self._assert_partition(dataset, list(splitter.split(dataset)))

    def test_invalid_fold_counts_are_rejected(self):
        for count in (0, 1, 2, 6, 3.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self._make_splitter(n_folds=count)

    def test_repeated_or_missing_subject_ids_are_rejected(self):
        for subjects in ([0, 0, 1, 2, 3], [], [0, 1, None]):
            with self.subTest(subjects=subjects), self.assertRaises(ValueError):
                self._make_splitter(all_subjects=subjects)

    def test_all_subjects_must_match_dataset(self):
        for subjects in ([0, 1, 2, 3, 99], [0, 1, 2]):
            with tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(ValueError, 'all_subjects'):
                    list(
                        self._make_splitter(os.path.join(path, 'split'),
                                            all_subjects=subjects,
                                            n_folds=3).split(_make_dataset()))

    def test_seeded_folds_are_reproducible_without_changing_global_rng(self):
        dataset = _make_dataset()
        np.random.seed(123)
        before = np.random.get_state()
        first = list(self._make_splitter().split(dataset))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        with tempfile.TemporaryDirectory() as path:
            repeated = list(
                self._make_splitter(os.path.join(path, 'split')).split(dataset))
        for a, b in zip(first, repeated):
            for x, y in zip(a, b):
                self.assertEqual(list(x.info['clip_id']), list(y.info['clip_id']))

    def test_changed_fold_configuration_is_rejected(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        with self.assertRaisesRegex(ValueError, 'parameters'):
            next(self._make_splitter(random_state=7).split(dataset))

    def test_unshuffled_folds_follow_supplied_subject_order(self):
        splitter = self._make_splitter(n_folds=3, shuffle=False)
        folds = list(splitter.split(_make_dataset()))
        self.assertEqual(set(folds[0][2].info['subject_id']), {0, 3})
        self.assertEqual(set(folds[0][1].info['subject_id']), {1, 4})
        self.assertEqual(set(folds[0][0].info['subject_id']), {2})

    def test_invalid_shuffle_and_random_state_are_rejected(self):
        for parameters in ({
                'shuffle': 'False'
        }, {
                'random_state': True
        }, {
                'random_state': -1
        }, {
                'random_state': 2**32
        }, {
                'random_state': 0.5
        }):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self._make_splitter(**parameters)


if __name__ == '__main__':
    unittest.main()
