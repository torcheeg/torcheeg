import unittest
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)

import mne
import numpy as np

from torcheeg.datasets import SleepEDFxDataset
from torcheeg.datasets.sleep_stage_detection import sleep_edfx


class TestSleepEDFxDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_sleep_edfx_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(SleepEDFxDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(SleepEDFxDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (2, 3000))

    def test_reader_preserves_declared_channel_order_with_missing_leads(self):
        cls = SleepEDFxDataset
        module = sleep_edfx
        names = cls.channels[::-1][:-1]
        data = np.arange(len(names) * 8).reshape(len(names), 8)
        raw = mne.io.RawArray(data, mne.create_info(
            names, 100, 'eeg'), verbose=False)
        selected = module._interpolate(raw, channels=cls.channels)
        self.assertEqual(selected.ch_names, cls.channels)
        expected = np.stack([data[names.index(name)] if name in names else np.zeros(
            8) for name in cls.channels])
        np.testing.assert_array_equal(selected.get_data(), expected)


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (SleepEDFxDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.data_class.fake_record(**kwargs))
                sample = next((item for item in _process_fake(
                    dataset_class, **kwargs) if 'info' in item))
                self.assertIsInstance(sample['info'], dict)
                self.assertNotIn('key', sample)
                self.assertNotIn('clip_id', sample['info'])
                self.assertGreater(sample['eeg'].size, 0)
                self.assertTrue(np.isfinite(sample['eeg']).all())

    def test_memory_construction_preserves_preprocessing_and_item_transforms(self):
        for dataset_class in (SleepEDFxDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(self, dataset_class)


if __name__ == '__main__':
    unittest.main()
