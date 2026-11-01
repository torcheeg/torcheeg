import unittest
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)

import numpy as np

from torcheeg.datasets import FACEDFeatureDataset


class TestFACEDFeatureDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_faced_feature_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(FACEDFeatureDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(FACEDFeatureDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (30, 5))


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (FACEDFeatureDataset,):
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
        for dataset_class in (FACEDFeatureDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(self, dataset_class)


if __name__ == '__main__':
    unittest.main()
