import unittest

from torcheeg.datasets import BCICIV2aDataset


class TestBciciv2A(unittest.TestCase):

    def test_bciciv_2a_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(BCICIV2aDataset.fake_record(**kwargs))
        item = next(BCICIV2aDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (22, 1750))


if __name__ == '__main__':
    unittest.main()
