import unittest

from torcheeg.datasets import KULDataset


class TestKul(unittest.TestCase):

    def test_kul_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULDataset.fake_record(**kwargs))
        item = next(KULDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))


if __name__ == '__main__':
    unittest.main()
