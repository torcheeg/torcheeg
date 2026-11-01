import unittest

from torcheeg.datasets import P2018Dataset


class TestP2018(unittest.TestCase):

    def test_p2018_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(P2018Dataset.fake_record(**kwargs))
        item = next(P2018Dataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (6, 3000))


if __name__ == '__main__':
    unittest.main()
