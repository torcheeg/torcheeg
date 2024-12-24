import unittest

from torcheeg.datasets import TSUBenckmarkDataset
from torcheeg.datasets import SanDiegoSSVEPDataset


class TestSSVEPDataset(unittest.TestCase):
    def test_tsu_benchmark_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(TSUBenckmarkDataset.fake_record(**kwargs))
        item = next(TSUBenckmarkDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 250))

    def test_sandiego_ssvep_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SanDiegoSSVEPDataset.fake_record(**kwargs))
        item = next(SanDiegoSSVEPDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (22, 256))


if __name__ == '__main__':
    unittest.main()
