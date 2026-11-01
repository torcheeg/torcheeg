import unittest

from torcheeg.datasets import KULProcessedDataset


class TestKulProcessed(unittest.TestCase):

    def test_kul_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULProcessedDataset.fake_record(**kwargs))
        item = next(KULProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))


if __name__ == '__main__':
    unittest.main()
