import unittest

import mne
import numpy as np

from torcheeg.datasets import HMCDataset
from torcheeg.datasets.sleep_stage_detection import hmc


class TestHmc(unittest.TestCase):

    def test_hmc_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(HMCDataset.fake_record(**kwargs))
        item = next(HMCDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (4, 3000))

    def test_reader_preserves_declared_channel_order_with_missing_leads(self):
        cls = HMCDataset
        module = hmc
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
