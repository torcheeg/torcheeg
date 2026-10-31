import unittest

from torcheeg import transforms
from torcheeg.datasets import SanDiegoSSVEPDataset, TSUBenckmarkDataset
from torcheeg.datasets.utils import get_locations


class TestSSVEPDataset(unittest.TestCase):
    def test_tsu_montage_matches_official_channel_labels(self):
        channels = TSUBenckmarkDataset.channels
        self.assertEqual([channels[i] for i in (32, 42, 59, 63)],
                         ['M1', 'M2', 'CB1', 'CB2'])
        self.assertEqual(set(TSUBenckmarkDataset.positions), set(channels))

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

        self.assertEqual(item['eeg'].shape, (8, 256))
        kwargs['offline_transform'] = transforms.ToGrid(get_locations(
            SanDiegoSSVEPDataset.channels, SanDiegoSSVEPDataset.grid))
        item = next(SanDiegoSSVEPDataset.process_record(**kwargs))
        self.assertEqual(item['eeg'].shape, (256, 2, 5))


if __name__ == '__main__':
    unittest.main()
