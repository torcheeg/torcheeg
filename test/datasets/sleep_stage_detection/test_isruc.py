import unittest

import mne
import numpy as np

from torcheeg.datasets import ISRUCDataset
from torcheeg.datasets.sleep_stage_detection import isruc


class TestIsruc(unittest.TestCase):

    def test_isruc_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(ISRUCDataset.fake_record(**kwargs))
        item = next(ISRUCDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (6, 3000))

    def test_reader_preserves_declared_channel_order_with_missing_leads(self):
        cls = ISRUCDataset
        module = isruc
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
