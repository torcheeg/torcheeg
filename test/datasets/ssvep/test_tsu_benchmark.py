import unittest

from torcheeg.datasets import TSUBenckmarkDataset
from torcheeg.datasets.utils import get_positions


class TestTsuBenchmark(unittest.TestCase):
    def test_tsu_montage_matches_official_channel_labels(self):
        channels = TSUBenckmarkDataset.channels
        self.assertEqual([channels[i] for i in (32, 42, 59, 63)],
                         ['M1', 'M2', 'CB1', 'CB2'])
        self.assertEqual(set(get_positions(channels)), set(channels))

    def test_tsu_benchmark_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(TSUBenckmarkDataset.fake_record(**kwargs))
        item = next(TSUBenckmarkDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 250))


if __name__ == '__main__':
    unittest.main()
