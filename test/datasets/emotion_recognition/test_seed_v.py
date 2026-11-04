import os
import unittest
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)

import numpy as np

from torcheeg.datasets import SEEDVDataset


class TestSEEDVDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_seed_v_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(SEEDVDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(SEEDVDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (62, 800))
        self.assertEqual(item['info']['date'], 20180507)
        self.assertIsInstance(item['info']['date'], int)

    def test_seed_v_metadata_from_record_names(self):
        samples = np.zeros((62, 40000))
        cases = [('10_1_20180507.cnt', 10, 1, 20180507, 4), (os.path.join(
            'nested', '01_2_20180607.cnt'), 1, 2, 20180607, 2), ('2_3_20180707.CNT', 2, 3, 20180707, 2)]
        for record, subject_id, session_id, date, emotion in cases:
            with self.subTest(record=record):
                item = next(_process_fake(SEEDVDataset, record, samples))
                info = item['info']
                for name, expected in (('subject_id', subject_id), ('session_id', session_id), ('date', date)):
                    self.assertEqual(info[name], expected)
                    self.assertIsInstance(info[name], int)
                self.assertEqual(info['emotion'], emotion)
                self.assertNotIn('clip_id', info)
                self.assertEqual(item['eeg'].shape, (62, 800))


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (SEEDVDataset,):
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
        for dataset_class in (SEEDVDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(self, dataset_class)


if __name__ == '__main__':
    unittest.main()
