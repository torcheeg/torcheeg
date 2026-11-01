import unittest

from torcheeg.datasets import AVEDProcessedDataset


class TestAvedProcessed(unittest.TestCase):

    def test_aved_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AVEDProcessedDataset.fake_record(**kwargs))
        item = next(AVEDProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
