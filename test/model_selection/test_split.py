import os
import tempfile
import unittest

from torcheeg.model_selection import (train_test_split,
                                      train_test_split_cross_trial,
                                      train_test_split_groupby_trial)

from .test_utils import _make_dataset


class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_train_test_split(self):
        dataset = _make_dataset()

        split_path = self._split_path

        train_dataset, test_dataset = train_test_split(dataset=dataset,
                                                       split_path=split_path,
                                                       test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)

    def test_train_test_split_cross_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path

        train_dataset, test_dataset = train_test_split_cross_trial(
            dataset=dataset, split_path=split_path, test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)

    def test_train_test_split_groupby_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path

        train_dataset, test_dataset = train_test_split_groupby_trial(
            dataset=dataset, split_path=split_path, test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)

    def test_groupby_trial_handles_different_trial_sets_per_subject(self):
        dataset = _make_dataset()
        dataset.info = dataset.info[~((dataset.info.subject_id == 0) &
                                      (dataset.info.trial_id == 1))].reset_index(
                                          drop=True)
        train, test = train_test_split_groupby_trial(dataset,
                                                     split_path=self._split_path)
        self.assertFalse(set(train.info.clip_id) & set(test.info.clip_id))
        self.assertEqual(
            set(train.info.clip_id) | set(test.info.clip_id), set(dataset.info.clip_id))
        self.assertEqual(set(train.info.groupby(['subject_id', 'trial_id']).groups),
                         set(test.info.groupby(['subject_id', 'trial_id']).groups))


if __name__ == '__main__':
    unittest.main()
