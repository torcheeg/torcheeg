import unittest

from torcheeg.datasets import SEEDVFeatureDataset


class TestSeedVFeature(unittest.TestCase):

    def test_seed_v_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDVFeatureDataset.fake_record(**kwargs))
        item = next(SEEDVFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))


if __name__ == '__main__':
    unittest.main()
