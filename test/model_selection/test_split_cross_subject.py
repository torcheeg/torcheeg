import os
import tempfile
import unittest

import numpy as np

from torcheeg.model_selection import train_test_split_cross_subject

from .test_utils import _make_dataset


class TestSplitCrossSubject(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self._path = os.path.join(directory.name, 'split')

    def test_subjects_are_isolated_without_losing_or_coercing_samples(self):
        dataset = _make_dataset()
        dataset.info['subject_id'] = dataset.info.subject_id.map(
            lambda value: f'{value:03}')
        train, test = train_test_split_cross_subject(
            dataset, test_size=0.4, split_path=self._path)
        self.assertFalse(set(train.info.subject_id)
                         & set(test.info.subject_id))
        self.assertEqual(set(train.info.subject_id), {'000', '001', '002'})
        self.assertEqual(set(test.info.subject_id), {'003', '004'})
        self.assertEqual(set(train.info.clip_id) | set(
            test.info.clip_id), set(dataset.info.clip_id))
        np.testing.assert_array_equal(test[0]['eeg'], dataset[75]['eeg'])
        before = {name: os.stat(os.path.join(
            self._path, name)).st_mtime_ns for name in os.listdir(self._path)}
        repeated = train_test_split_cross_subject(
            dataset, test_size=0.4, split_path=self._path)
        self.assertTrue(repeated[1].info.equals(test.info))
        self.assertEqual(before, {name: os.stat(os.path.join(
            self._path, name)).st_mtime_ns for name in os.listdir(self._path)})

    def test_seeded_stratification_keeps_both_subject_classes(self):
        dataset = _make_dataset()
        dataset.info = dataset.info[dataset.info.subject_id.lt(4)].copy()
        outcomes = []
        for suffix in ('a', 'b'):
            train, test = train_test_split_cross_subject(
                dataset, test_size=0.5, shuffle=True, random_state=3, label_transform=lambda y: y['subject_id'] % 2, split_path=self._path + suffix)
            for subset in (train, test):
                self.assertEqual(set(subset.info.subject_id % 2), {0, 1})
            outcomes.append(test.info.clip_id.tolist())
        self.assertEqual(outcomes[0], outcomes[1])

    def test_invalid_size_and_changed_parameters_are_rejected(self):
        dataset = _make_dataset()
        with self.assertRaises(ValueError):
            train_test_split_cross_subject(
                dataset, test_size=1.0, split_path=self._path)
        train_test_split_cross_subject(
            dataset, test_size=0.2, split_path=self._path)
        with self.assertRaises(ValueError):
            train_test_split_cross_subject(
                dataset, test_size=0.4, split_path=self._path)


if __name__ == '__main__':
    unittest.main()
