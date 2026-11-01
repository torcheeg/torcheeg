import unittest

from torcheeg.datasets import MPEDFeatureDataset


class TestMpedFeature(unittest.TestCase):

    def test_mped_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(MPEDFeatureDataset.fake_record(**kwargs))
        item = next(MPEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 23))


if __name__ == '__main__':
    unittest.main()
