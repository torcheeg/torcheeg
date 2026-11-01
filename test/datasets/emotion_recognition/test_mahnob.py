import unittest

from torcheeg.datasets import MAHNOBDataset


class TestMahnob(unittest.TestCase):

    def test_mahnob_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(MAHNOBDataset.fake_record(**kwargs))
        item = next(MAHNOBDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
