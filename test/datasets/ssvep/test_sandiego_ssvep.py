import unittest

from torcheeg import transforms
from torcheeg.datasets import SanDiegoSSVEPDataset
from torcheeg.datasets.utils import get_locations


class TestSandiegoSsvep(unittest.TestCase):

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
