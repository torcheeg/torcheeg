import unittest

import numpy as np

from torcheeg.datasets import SEEDIVDataset


class TestSEEDIVDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_seed_iv_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDIVDataset.fake_record(**kwargs))
        item = next(SEEDIVDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 800))


class TestFakeRecordMetadata(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (SEEDIVDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.fake_record(**kwargs))
                sample = next(item for item in dataset_class.process_record(**kwargs)
                              if 'info' in item)
                self.assertIsInstance(sample['key'], str)
                self.assertEqual(sample['info']['clip_id'], sample['key'])
                self.assertGreater(sample['eeg'].size, 0)
                self.assertTrue(np.isfinite(sample['eeg']).all())


if __name__ == '__main__':
    unittest.main()
