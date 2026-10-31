import os
import tempfile
import unittest

from torcheeg.model_selection import TrainValTestSplitCrossSubject

from .test_utils import _make_dataset


class TestTrainValTestSplitCrossSubject(unittest.TestCase):
    _splitter_class = TrainValTestSplitCrossSubject
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

    def test_train_val_test_split_cross_subject(self):
        train, val, test = next(self._make_splitter().split(_make_dataset()))
        self.assertEqual([len(train), len(val), len(test)], [75, 25, 25])

    def test_overlapping_empty_duplicate_and_missing_partition_ids_are_rejected(self):
        for ids in ([0, 1, 3], [], [0, 0, 1], [None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self._make_splitter(train_subjects=ids)

    def test_unknown_partition_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            list(self._make_splitter(train_subjects=[
                 0, 1, 99]).split(_make_dataset()))

    def test_unassigned_ids_can_be_intentionally_excluded(self):
        train, val, test = next(
            self._make_splitter(train_subjects=[0], val_subjects=[1],
                                test_subjects=[2]).split(_make_dataset()))
        self.assertEqual([len(train), len(val), len(test)], [25, 25, 25])

    def test_changed_partition_configuration_does_not_reuse_old_cache(self):
        dataset = _make_dataset()
        list(self._make_splitter().split(dataset))
        with self.assertRaisesRegex(ValueError, 'parameters'):
            next(
                self._make_splitter(train_subjects=[4],
                                    test_subjects=[0, 1, 2]).split(dataset))

    def test_string_ids_keep_their_values_and_types(self):
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info['subject_id'].map(
            lambda value: f'id_{value:02d}')
        splitter = self._make_splitter(train_subjects=['id_00', 'id_01', 'id_02'],
                                       val_subjects=['id_03'],
                                       test_subjects=['id_04'])
        folds = list(splitter.split(dataset))
        self._assert_partition(dataset, folds)
        self.assertEqual(set(folds[0][1].info['subject_id']), {'id_03'})

    def test_caller_id_lists_are_copied(self):
        ids = [0, 1, 2]
        splitter = self._make_splitter(train_subjects=ids)
        ids.append(3)
        self.assertEqual(splitter.train_subjects, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
