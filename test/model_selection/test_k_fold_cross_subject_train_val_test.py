import os
import tempfile
import unittest

import numpy as np

from torcheeg.model_selection import KFoldCrossSubjectTrainValTest

from ._three_way_cases import ThreeWayChecks, make_three_way_dataset


class TestKFoldCrossSubjectTrainValTest(ThreeWayChecks, unittest.TestCase):
    splitter_class = KFoldCrossSubjectTrainValTest
    parameters = {'all_subjects': [0, 1, 2, 3, 4], 'n_folds': 5}
    fold_count = 5
    group_column = 'subject_id'
    metadata_column = 'subject_id'

    def test_k_fold_cross_subject_train_val_test(self):
        folds = list(self.make_splitter().split(make_three_way_dataset()))
        self.assertEqual(len(folds), 5)
        validation_subjects, test_subjects = [], []
        for train, val, test in folds:
            self.assertEqual([len(train), len(val), len(test)], [75, 25, 25])
            validation_subjects.extend(val.info['subject_id'].unique())
            test_subjects.extend(test.info['subject_id'].unique())
        self.assertCountEqual(validation_subjects, list(range(5)))
        self.assertCountEqual(test_subjects, list(range(5)))

    def test_string_subject_ids_keep_their_values_and_types(self):
        dataset = make_three_way_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map(lambda value: f'sub_{value:02d}')
        splitter = self.make_splitter(all_subjects=[f'sub_{value:02d}' for value in range(5)])
        self.assert_partition(dataset, list(splitter.split(dataset)))

    def test_invalid_fold_counts_are_rejected(self):
        for count in (0, 1, 2, 6, 3.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.make_splitter(n_folds=count)

    def test_repeated_or_missing_subject_ids_are_rejected(self):
        for subjects in ([0, 0, 1, 2, 3], [], [0, 1, None]):
            with self.subTest(subjects=subjects), self.assertRaises(ValueError):
                self.make_splitter(all_subjects=subjects)

    def test_all_subjects_must_match_dataset(self):
        for subjects in ([0, 1, 2, 3, 99], [0, 1, 2]):
            with tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(ValueError, 'all_subjects'):
                    list(self.make_splitter(os.path.join(path, 'split'),
                        all_subjects=subjects, n_folds=3).split(make_three_way_dataset()))

    def test_seeded_folds_are_reproducible_without_changing_global_rng(self):
        dataset = make_three_way_dataset()
        np.random.seed(123)
        before = np.random.get_state()
        first = list(self.make_splitter().split(dataset))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        with tempfile.TemporaryDirectory() as path:
            repeated = list(self.make_splitter(os.path.join(path, 'split')).split(dataset))
        for a, b in zip(first, repeated):
            for x, y in zip(a, b):
                self.assertEqual(list(x.info['clip_id']), list(y.info['clip_id']))

    def test_changed_fold_configuration_is_rejected(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        with self.assertRaisesRegex(ValueError, 'parameters'):
            next(self.make_splitter(random_state=7).split(dataset))

    def test_unshuffled_folds_follow_supplied_subject_order(self):
        splitter = self.make_splitter(n_folds=3, shuffle=False)
        folds = list(splitter.split(make_three_way_dataset()))
        self.assertEqual(set(folds[0][2].info['subject_id']), {0, 3})
        self.assertEqual(set(folds[0][1].info['subject_id']), {1, 4})
        self.assertEqual(set(folds[0][0].info['subject_id']), {2})


if __name__ == '__main__':
    unittest.main()
