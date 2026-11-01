import unittest

from torcheeg.datasets import FACEDFeatureDataset


class TestFacedFeature(unittest.TestCase):

    def test_faced_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(FACEDFeatureDataset.fake_record(**kwargs))
        item = next(FACEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 5))


if __name__ == '__main__':
    unittest.main()
