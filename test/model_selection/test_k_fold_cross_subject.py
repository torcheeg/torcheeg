import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldCrossSubject

from .test_utils import _make_dataset


class TestKFoldCrossSubject(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_k_fold_cross_subject(self):
        dataset = _make_dataset()

        split_path = self._split_path
        k_fold = KFoldCrossSubject(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_subject_folds_follow_first_occurrence_order(self):
        dataset = _make_dataset()
        subjects = ['subject_z', 'subject_a',
                    'subject_m', 'subject_b', 'subject_y']
        dataset.info['subject_id'] = dataset.info['subject_id'].map(
            dict(enumerate(subjects)))
        splitter = KFoldCrossSubject(n_splits=5, split_path=self._split_path)
        folds = list(splitter.split(dataset))
        self.assertEqual(
            [test.info['subject_id'].unique().tolist() for _, test in folds],
            [[subject] for subject in subjects])


if __name__ == '__main__':
    unittest.main()
