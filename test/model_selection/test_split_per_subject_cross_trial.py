import os
import tempfile
import unittest

from torcheeg.model_selection import train_test_split_per_subject_cross_trial

from .test_utils import _make_dataset


class TestSplitPerSubjectCrossTrial(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_train_test_split_per_subject_cross_trial(self):
        dataset = _make_dataset()
        split_path = self._split_path
        train_dataset, test_dataset = train_test_split_per_subject_cross_trial(
            dataset=dataset, split_path=split_path, subject=0)
        self.assertEqual(len(train_dataset), 20)
        self.assertEqual(len(test_dataset), 5)


if __name__ == '__main__':
    unittest.main()
