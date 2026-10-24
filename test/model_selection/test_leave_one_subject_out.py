import os
import tempfile
import unittest

from torcheeg.model_selection import LeaveOneSubjectOut

from .test_utils import _make_dataset


class TestLeaveOneSubjectOut(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def test_leave_one_subject_out(self):
        dataset = _make_dataset()

        split_path = self._split_path
        k_fold = LeaveOneSubjectOut(split_path=split_path)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 100)
            self.assertEqual(len(test_dataset), 25)
            split_num += 1
        self.assertEqual(split_num, 5)


if __name__ == '__main__':
    unittest.main()
