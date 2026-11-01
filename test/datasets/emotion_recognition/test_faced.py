import unittest

from torcheeg.datasets import FACEDDataset


class TestFaced(unittest.TestCase):

    def test_faced_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(FACEDDataset.fake_record(**kwargs))
        item = next(FACEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 250))


if __name__ == '__main__':
    unittest.main()
