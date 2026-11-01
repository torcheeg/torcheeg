import unittest

from torcheeg.datasets import DTUProcessedDataset


class TestDtuProcessed(unittest.TestCase):

    def test_dtu_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUProcessedDataset.fake_record(**kwargs))
        item = next(DTUProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))


if __name__ == '__main__':
    unittest.main()
