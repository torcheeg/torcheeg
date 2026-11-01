import os
import tempfile
import unittest

from torcheeg.datasets import DEAPDataset


class TestInputValidation(unittest.TestCase):

    def test_missing_dataset_directory_is_a_file_error(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(FileNotFoundError):
                DEAPDataset(root_path=os.path.join(path, 'missing'),
                            io_path=os.path.join(path, 'cache'), verbose=False)


class TestDeap(unittest.TestCase):

    def test_deap_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DEAPDataset.fake_record(**kwargs))
        item = next(DEAPDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
