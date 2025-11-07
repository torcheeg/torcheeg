import unittest

from torcheeg.datasets import DTUDataset, DTUProcessedDataset, KULDataset, KULProcessedDataset, AVEDDataset, AVEDProcessedDataset


class TestAuditoryAttention(unittest.TestCase):

    def test_dtu_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUDataset.fake_record(**kwargs))
        item = next(DTUDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_dtu_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUProcessedDataset.fake_record(**kwargs))
        item = next(DTUProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_kul_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULDataset.fake_record(**kwargs))
        item = next(KULDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_kul_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULProcessedDataset.fake_record(**kwargs))
        item = next(KULProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_aved_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AVEDDataset.fake_record(**kwargs))
        item = next(AVEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))

    def test_aved_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AVEDProcessedDataset.fake_record(**kwargs))
        item = next(AVEDProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
