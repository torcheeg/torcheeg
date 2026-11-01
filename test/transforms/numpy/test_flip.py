import unittest

import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_locations
from torcheeg.transforms import HorizontalFlip


class TestFlipTransforms(unittest.TestCase):

    def test_horizontal_flip(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = HorizontalFlip(
            location_dict=get_locations(DEAPDataset.channels, DEAPDataset.grid), channel_dim=0)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
