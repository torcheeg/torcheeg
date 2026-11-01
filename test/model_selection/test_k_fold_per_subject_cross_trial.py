import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldPerSubjectCrossTrial

from .test_utils import _make_dataset


class TestKFoldPerSubjectCrossTrial(unittest.TestCase):

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
            self.assertEqual(
                len(list(set(train_dataset.info['subject_id']))), 1)
            self.assertEqual(
                len(list(set(test_dataset.info['subject_id']))), 1)
            split_num += 1
        self.assertEqual(split_num, 25)


if __name__ == '__main__':
    unittest.main()
