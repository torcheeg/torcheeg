import os
import tempfile
import unittest

from torcheeg.model_selection import train_test_split

from .test_utils import _make_dataset


class TestSplit(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
