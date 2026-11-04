import unittest
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)
from unittest.mock import patch

import numpy as np

from torcheeg import transforms
from torcheeg.datasets import SEEDDataset, SEEDFeatureDataset


class TestSEEDDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_seed_class_metadata_before_initialization(self):
        from torcheeg.datasets import CachedDataset
        from torcheeg.datasets.utils import get_locations, get_positions
        self.assertEqual(len(SEEDDataset.channels), 62)
        self.assertEqual(SEEDDataset.channels[:3], ['FP1', 'FPZ', 'FP2'])
        self.assertEqual(set(get_positions(SEEDDataset.channels)),
                         set(SEEDDataset.channels))
        transform = transforms.ToGrid(get_locations(
            SEEDDataset.channels, SEEDDataset.grid))
        with patch.object(CachedDataset, '__init__', return_value=None):
            dataset = SEEDDataset(offline_transform=transform)
            SEEDDataset(num_channel=32)
        self.assertIs(dataset.offline_transform, transform)
        self.assertEqual(len(SEEDDataset.channels), 62)
        self.assertEqual(SEEDFeatureDataset.channels, SEEDDataset.channels)
        self.assertEqual(SEEDFeatureDataset.grid, SEEDDataset.grid)
        self.assertEqual(SEEDFeatureDataset.adjacency, SEEDDataset.adjacency)

    def test_seed_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(SEEDDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(SEEDDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (62, 200))


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (SEEDDataset,):
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
        for dataset_class in (SEEDDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(self, dataset_class)


if __name__ == '__main__':
    unittest.main()
