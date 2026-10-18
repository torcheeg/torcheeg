import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from torcheeg import model_selection

from .test_utils import _make_dataset


class TestBaseModelSelection(unittest.TestCase):
    _parameters = {
        'train_subjects': [0, 1, 2],
        'val_subjects': [3],
        'test_subjects': [4]
    }

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_base_methods_require_subclass_implementation(self):
        base = model_selection.BaseModelSelection(split_path=self._split_path)
        with self.assertRaisesRegex(NotImplementedError, 'Method build'):
            base.build(_make_dataset().info)
        with self.assertRaisesRegex(NotImplementedError, 'Method split'):
            base.split(_make_dataset())
        self.assertEqual(
            repr(base),
            f'BaseModelSelection(split_path={self._split_path!r}, timeout=300.0)')

    def test_existing_and_new_splitters_use_the_shared_index_cache(self):
        dataset = _make_dataset()
        dataset.info['trial_id'] = dataset.info['trial_id'].map(lambda value:
                                                                (value + 1) * 10)
        classes = [
            ('KFold', {
                'n_splits': 5
            }),
            ('KFoldCrossSubject', {
                'n_splits': 5
            }),
            ('KFoldCrossTrial', {
                'n_splits': 5
            }),
            ('KFoldGroupbyTrial', {
                'n_splits': 5
            }),
            ('KFoldPerSubject', {
                'n_splits': 5
            }),
            ('KFoldPerSubjectCrossTrial', {
                'n_splits': 5
            }),
            ('KFoldPerSubjectGroupbyTrial', {
                'n_splits': 5
            }),
            ('LeaveOneSubjectOut', {}),
            ('Subcategory', {
                'criteria': 'subject_id'
            }),
            ('TrainValTestSplitCrossSubject', self._parameters),
            ('TrainValTestSplitCrossTrial', {
                'train_trials': [10, 20, 30],
                'val_trials': [40],
                'test_trials': [50]
            }),
            ('TrainValTestSplitGroupbyFolder', {}),
            ('KFoldCrossSubjectTrainValTest', {
                'all_subjects': list(range(5))
            }),
            ('TrainTestSplitPerSubjectGroupbyTrial', {}),
        ]
        functions = [
            ('train_test_split', {}),
            ('train_test_split_cross_subject', {}),
            ('train_test_split_cross_trial', {}),
            ('train_test_split_groupby_trial', {}),
            ('train_test_split_per_subject_cross_trial', {
                'subject': 0
            }),
            ('train_test_split_per_subject_groupby_trial', {
                'subject': 0
            }),
        ]
        for name, parameters in classes + functions:
            with self.subTest(
                    splitter=name), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, 'split')
                target = getattr(model_selection, name)
                if name[0].isupper():
                    splitter = target(split_path=path, **parameters)
                    self.assertIsInstance(splitter, model_selection.BaseModelSelection)
                    run = lambda: list(splitter.split(dataset))
                else:
                    run = lambda: [target(dataset, split_path=path, **parameters)]
                original = run()
                repeated = run()
                self.assertGreater(len(original), 0)
                self.assertEqual(len(original), len(repeated))
                covered = set()
                for first, second in zip(original, repeated):
                    first = first if isinstance(first, tuple) else (first, )
                    second = second if isinstance(second, tuple) else (second, )
                    self.assertEqual(len(first), len(second))
                    clips = [set(subset.info['clip_id']) for subset in first]
                    covered.update(set.union(*clips))
                    for index, left in enumerate(first):
                        for right in first[index + 1:]:
                            self.assertFalse(
                                set(left.info.clip_id) & set(right.info.clip_id))
                            if name in ('KFoldCrossSubject', 'LeaveOneSubjectOut',
                                        'TrainValTestSplitCrossSubject',
                                        'KFoldCrossSubjectTrainValTest',
                                        'train_test_split_cross_subject'):
                                self.assertFalse(
                                    set(left.info.subject_id)
                                    & set(right.info.subject_id))
                            if name in ('KFoldCrossTrial',
                                        'TrainValTestSplitCrossTrial'):
                                self.assertFalse(
                                    set(left.info.trial_id) & set(right.info.trial_id))
                            left_groups = set(
                                zip(left.info.subject_id, left.info.trial_id))
                            right_groups = set(
                                zip(right.info.subject_id, right.info.trial_id))
                            if name in ('KFoldPerSubjectCrossTrial',
                                        'train_test_split_cross_trial',
                                        'train_test_split_per_subject_cross_trial'):
                                self.assertFalse(left_groups & right_groups)
                            if name in ('KFoldGroupbyTrial',
                                        'KFoldPerSubjectGroupbyTrial',
                                        'TrainTestSplitPerSubjectGroupbyTrial',
                                        'train_test_split_groupby_trial',
                                        'train_test_split_per_subject_groupby_trial'):
                                self.assertEqual(left_groups, right_groups)
                    for subset, cached in zip(first, second):
                        pd.testing.assert_frame_equal(subset.info, cached.info)
                        positions = subset.info['clip_id'].astype(int).tolist()
                        expected = dataset.info.iloc[positions].reset_index(drop=True)
                        pd.testing.assert_frame_equal(subset.info, expected)
                        np.testing.assert_array_equal(subset[0][0],
                                                      dataset[positions[0]][0])
                expected_rows = dataset.info
                if name in ('train_test_split_per_subject_cross_trial',
                            'train_test_split_per_subject_groupby_trial'):
                    expected_rows = expected_rows[expected_rows.subject_id == 0]
                self.assertEqual(covered, set(expected_rows.clip_id))
                # The same public operation must reject a different row order.
                dataset.info = dataset.info.iloc[::-1].reset_index(drop=True)
                try:
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        run()
                finally:
                    dataset.info = dataset.info.iloc[::-1].reset_index(drop=True)
