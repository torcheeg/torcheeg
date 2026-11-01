import unittest

from torcheeg.datasets import SEEDIVDataset


class TestSeedIv(unittest.TestCase):

    def test_seed_iv_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDIVDataset.fake_record(**kwargs))
        item = next(SEEDIVDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 800))


if __name__ == '__main__':
    unittest.main()
