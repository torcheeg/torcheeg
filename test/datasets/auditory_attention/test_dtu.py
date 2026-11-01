import unittest

from torcheeg.datasets import DTUDataset


class TestDtu(unittest.TestCase):

    def test_dtu_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUDataset.fake_record(**kwargs))
        item = next(DTUDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))


if __name__ == '__main__':
    unittest.main()
