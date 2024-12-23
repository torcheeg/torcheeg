import unittest

from torcheeg.datasets import M3CVDataset

class TestPersonalIdentification(unittest.TestCase):

    def test_bciciv_2a_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(M3CVDataset.fake_record(**kwargs))
        item = next(M3CVDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 1000))


if __name__ == '__main__':
    unittest.main()
