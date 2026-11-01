import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldCrossTrial

from .test_utils import _make_dataset


class TestKFoldCrossTrial(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_k_fold_cross_trial(self):
        dataset = _make_dataset()

        split_path = self._split_path
        k_fold = KFoldCrossTrial(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_mixed_trial_id_types_are_not_coerced(self):
        dataset = _make_dataset()
        trials = [10, 'trial_a', 30, 'trial_b', 50]
        dataset.info['trial_id'] = dataset.info['trial_id'].map(
            dict(enumerate(trials)))
        splitter = KFoldCrossTrial(n_splits=5, split_path=self._split_path)
        folds = list(splitter.split(dataset))
        self.assertEqual([test.info['trial_id'].unique().tolist() for _, test in folds],
                         [[trial] for trial in trials])


if __name__ == '__main__':
    unittest.main()
