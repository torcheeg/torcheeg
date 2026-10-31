import unittest

import mne
import numpy as np

from torcheeg.datasets import (HMCDataset, ISRUCDataset, P2018Dataset,
                               SleepEDFxDataset)
from torcheeg.datasets.sleep_stage_detection import hmc, isruc, sleep_edfx


class TestSleepStageDetectionDataset(unittest.TestCase):

    def test_readers_preserve_declared_channel_order_with_missing_leads(self):
        for cls, module in ((HMCDataset, hmc), (ISRUCDataset, isruc),
                            (SleepEDFxDataset, sleep_edfx)):
            with self.subTest(dataset=cls.__name__):
                names = cls.channels[::-1][:-1]
                data = np.arange(len(names) * 8).reshape(len(names), 8)
                raw = mne.io.RawArray(data, mne.create_info(names, 100, 'eeg'),
                                      verbose=False)
                selected = module._interpolate(raw, channels=cls.channels)
                self.assertEqual(selected.ch_names, cls.channels)
                expected = np.stack([data[names.index(name)] if name in names
                                     else np.zeros(8) for name in cls.channels])
                np.testing.assert_array_equal(selected.get_data(), expected)

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
