import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldGroupbyTrial

from .test_utils import _make_dataset


class TestKFoldGroupbyTrial(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_kfold_groupby_trial(self):
        dataset = _make_dataset()
        split_path = self._split_path
        k_fold = KFoldGroupbyTrial(split_path=split_path, n_splits=5)
        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)


if __name__ == '__main__':
    unittest.main()
