import unittest

import mne
import numpy as np

from torcheeg.datasets import SleepEDFxDataset
from torcheeg.datasets.sleep_stage_detection import sleep_edfx


class TestSleepEdfx(unittest.TestCase):

    def test_sleep_edfx_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SleepEDFxDataset.fake_record(**kwargs))
        item = next(SleepEDFxDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (2, 3000))

    def test_reader_preserves_declared_channel_order_with_missing_leads(self):
        cls = SleepEDFxDataset
        module = sleep_edfx
        names = cls.channels[::-1][:-1]
        data = np.arange(len(names) * 8).reshape(len(names), 8)
        raw = mne.io.RawArray(data, mne.create_info(names, 100, 'eeg'),
                              verbose=False)
        selected = module._interpolate(raw, channels=cls.channels)
        self.assertEqual(selected.ch_names, cls.channels)
        expected = np.stack([data[names.index(name)] if name in names
                             else np.zeros(8) for name in cls.channels])
        np.testing.assert_array_equal(selected.get_data(), expected)


if __name__ == '__main__':
    unittest.main()
