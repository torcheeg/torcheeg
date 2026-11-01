import unittest

from torcheeg import transforms
from torcheeg.datasets import SEEDDataset, SEEDFeatureDataset


class TestSeed(unittest.TestCase):

    def test_seed_class_metadata_before_initialization(self):
        from unittest.mock import patch

        from torcheeg.datasets import BaseDataset
        from torcheeg.datasets.utils import get_locations, get_positions

        self.assertEqual(len(SEEDDataset.channels), 62)
        self.assertEqual(SEEDDataset.channels[:3], ['FP1', 'FPZ', 'FP2'])
        self.assertEqual(set(get_positions(SEEDDataset.channels)),
                         set(SEEDDataset.channels))
        transform = transforms.ToGrid(get_locations(
            SEEDDataset.channels, SEEDDataset.grid))
        with patch.object(BaseDataset, '__init__', return_value=None):
            dataset = SEEDDataset(offline_transform=transform)
            SEEDDataset(num_channel=32)
        self.assertIs(dataset.offline_transform, transform)
        self.assertEqual(len(SEEDDataset.channels), 62)
        self.assertEqual(SEEDFeatureDataset.channels, SEEDDataset.channels)
        self.assertEqual(SEEDFeatureDataset.grid, SEEDDataset.grid)
        self.assertEqual(SEEDFeatureDataset.adjacency, SEEDDataset.adjacency)

    def test_seed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDDataset.fake_record(**kwargs))
        item = next(SEEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 200))


if __name__ == '__main__':
    unittest.main()
