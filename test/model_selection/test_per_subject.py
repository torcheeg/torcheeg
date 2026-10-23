import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldPerSubjectCrossTrial, KFoldPerSubjectGroupbyTrial, train_test_split_per_subject_cross_trial, train_test_split_per_subject_groupby_trial

from .test_utils import _make_dataset


class TestPerSubjectSplits(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_k_fold_per_subject_cross_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path
        k_fold = KFoldPerSubjectCrossTrial(split_path=split_path, n_splits=5)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 20)
            self.assertEqual(len(test_dataset), 5)
            self.assertEqual(len(list(set(train_dataset.info['subject_id']))), 1)
            self.assertEqual(len(list(set(test_dataset.info['subject_id']))), 1)
            split_num += 1
        self.assertEqual(split_num, 25)

    def test_k_fold_per_subject_groupby_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path
        k_fold = KFoldPerSubjectGroupbyTrial(split_path=split_path, n_splits=5)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 20)
            self.assertEqual(len(test_dataset), 5)
            self.assertEqual(len(list(set(train_dataset.info['subject_id']))), 1)
            self.assertEqual(len(list(set(test_dataset.info['subject_id']))), 1)
            split_num += 1
        self.assertEqual(split_num, 25)

    def test_train_test_split_per_subject_cross_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path

        train_dataset, test_dataset = train_test_split_per_subject_cross_trial(
            dataset=dataset, split_path=split_path, subject=0)

        self.assertEqual(len(train_dataset), 20)
        self.assertEqual(len(test_dataset), 5)

    def test_train_test_split_per_subject_groupby_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path

        train_dataset, test_dataset = train_test_split_per_subject_groupby_trial(
            dataset=dataset, split_path=split_path, subject=0)

        self.assertEqual(len(train_dataset), 20)
        self.assertEqual(len(test_dataset), 5)

    def test_numeric_and_string_subject_selectors(self):
        from torcheeg import model_selection
        for name in ('KFoldPerSubject', 'KFoldPerSubjectCrossTrial',
                     'KFoldPerSubjectGroupbyTrial'):
            for subject in (1, '1'):
                with self.subTest(
                        splitter=name,
                        subject=subject), tempfile.TemporaryDirectory() as path:
                    splitter = getattr(model_selection, name)(n_splits=5,
                                                              split_path=path)
                    folds = list(splitter.split(_make_dataset(), subject=subject))
                    self.assertEqual(len(folds), 5)
                    for train, test in folds:
                        self.assertEqual(set(train.info['subject_id']), {1})
                        self.assertEqual(set(test.info['subject_id']), {1})
                    with self.assertRaisesRegex(ValueError, 'subject'):
                        list(splitter.split(_make_dataset(), subject=99))

    def test_subject_ids_with_path_separators_are_round_tripped(self):
        from torcheeg import model_selection
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map(
            lambda value: f'group/{value}%')
        for name in ('KFoldPerSubject', 'KFoldPerSubjectCrossTrial',
                     'KFoldPerSubjectGroupbyTrial', 'LeaveOneSubjectOut',
                     'TrainTestSplitPerSubjectGroupbyTrial'):
            with self.subTest(splitter=name), tempfile.TemporaryDirectory() as path:
                kwargs = {} if name in ('LeaveOneSubjectOut',
                                        'TrainTestSplitPerSubjectGroupbyTrial') else {
                                            'n_splits': 5
                                        }
                splitter = getattr(model_selection, name)(split_path=path, **kwargs)
                folds = list(splitter.split(dataset, subject='group/1%'))
                self.assertEqual(len(folds), 5 if 'KFold' in name else 1)
                self.assertIn('group/1%', splitter.subjects)
                held_out = folds[0][1] if name == 'LeaveOneSubjectOut' else folds[0][0]
                self.assertEqual(set(held_out.info['subject_id']), {'group/1%'})

    def test_ambiguous_subject_filename_ids_are_rejected(self):
        from torcheeg import model_selection
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map({
            0: 0,
            1: '0',
            2: 2,
            3: 3,
            4: 4
        })
        for name in ('KFoldPerSubject', 'KFoldPerSubjectCrossTrial',
                     'KFoldPerSubjectGroupbyTrial', 'LeaveOneSubjectOut',
                     'TrainTestSplitPerSubjectGroupbyTrial'):
            with self.subTest(splitter=name), tempfile.TemporaryDirectory() as path:
                splitter = getattr(model_selection, name)(split_path=path)
                with self.assertRaisesRegex(ValueError, 'distinct string'):
                    list(splitter.split(dataset))


if __name__ == '__main__':
    unittest.main()
