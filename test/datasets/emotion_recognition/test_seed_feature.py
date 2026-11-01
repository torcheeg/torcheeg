import unittest

from torcheeg.datasets import SEEDFeatureDataset


class TestSeedFeature(unittest.TestCase):

    def test_seed_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDFeatureDataset.fake_record(**kwargs))
        item = next(SEEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))


if __name__ == '__main__':
    unittest.main()
