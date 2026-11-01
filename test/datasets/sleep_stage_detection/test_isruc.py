import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.datasets import ISRUCDataset
from torcheeg.datasets.sleep_stage_detection import isruc


class TestISRUCDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

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


class TestFakeRecordMetadata(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (ISRUCDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.fake_record(**kwargs))
                sample = next(item for item in dataset_class.process_record(**kwargs)
                              if 'info' in item)
                self.assertIsInstance(sample['key'], str)
                self.assertEqual(sample['info']['clip_id'], sample['key'])
                self.assertGreater(sample['eeg'].size, 0)
                self.assertTrue(np.isfinite(sample['eeg']).all())

    def test_memory_construction_preserves_preprocessing_and_item_transforms(self):
        for dataset_class in (ISRUCDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                payload = dataset_class.fake_record(record='fake_record')
                record = payload.pop('record', 'fake_record')
                original_process = dataset_class.process_record
                expected = []

                def process_one_sample(**kwargs):
                    for item in original_process(**kwargs):
                        yield item
                        if 'info' in item:
                            expected.append(item['eeg'].copy())
                            break

                def increment(**signals):
                    return {name: value + 1 for name, value in signals.items()}

                with patch.object(dataset_class, 'set_records', return_value=[record]), patch.object(
                        dataset_class, 'read_record', return_value=payload), patch.object(
                        dataset_class, 'process_record', side_effect=process_one_sample):
                    dataset = dataset_class(
                        io_mode='memory', verbose=False,
                        before_trial=lambda signal: signal * 2,
                        offline_transform=increment,
                        online_transform=lambda **signals: {
                            'eeg': signals['eeg'] + 3},
                        label_transform=lambda y: {'y': y['clip_id']})
                self.assertEqual(len(dataset), 1)
                signal, label = dataset[0]
                np.testing.assert_array_equal(signal, expected[0] + 3)
                self.assertEqual(label, dataset.info.iloc[0]['clip_id'])
                self.assertIn('io_mode', dataset.repr_body)
                self.assertIn(dataset_class.__name__, repr(dataset))
                dataset.online_transform = None
                dataset.label_transform = None
                signal, info = dataset[0]
                np.testing.assert_array_equal(signal, expected[0])
                self.assertEqual(info['clip_id'], label)


if __name__ == '__main__':
    unittest.main()
