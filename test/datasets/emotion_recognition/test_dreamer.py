import unittest

from torcheeg.datasets import DREAMERDataset


class TestDreamer(unittest.TestCase):

    def test_dreamer_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DREAMERDataset.fake_record(**kwargs))
        item = next(DREAMERDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (14, 128))


if __name__ == '__main__':
    unittest.main()
