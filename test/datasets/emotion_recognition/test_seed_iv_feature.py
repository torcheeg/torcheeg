import unittest

from torcheeg.datasets import SEEDIVFeatureDataset


class TestSeedIvFeature(unittest.TestCase):

    def test_seed_iv_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDIVFeatureDataset.fake_record(**kwargs))
        item = next(SEEDIVFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))


if __name__ == '__main__':
    unittest.main()
