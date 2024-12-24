import unittest

from torcheeg.datasets import SleepEDFxDataset, HMCDataset, ISRUCDataset, P2018Dataset


class TestSleepStageDetectionDataset(unittest.TestCase):

    def test_sleep_edfx_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SleepEDFxDataset.fake_record(**kwargs))
        item = next(SleepEDFxDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (2, 3000))

    def test_hmc_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(HMCDataset.fake_record(**kwargs))
        item = next(HMCDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (4, 3000))

    def test_isruc_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(ISRUCDataset.fake_record(**kwargs))
        item = next(ISRUCDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (6, 3000))

    def test_p2018_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(P2018Dataset.fake_record(**kwargs))
        item = next(P2018Dataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (6, 3000))


if __name__ == '__main__':
    unittest.main()
