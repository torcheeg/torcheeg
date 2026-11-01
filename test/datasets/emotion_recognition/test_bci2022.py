import unittest

from torcheeg.datasets import BCI2022Dataset


class TestBci2022(unittest.TestCase):

    def test_bci2022(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(BCI2022Dataset.fake_record(**kwargs))
        item = next(BCI2022Dataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 250))


if __name__ == '__main__':
    unittest.main()
