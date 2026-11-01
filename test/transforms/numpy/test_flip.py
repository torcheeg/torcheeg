import unittest

import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_locations
from torcheeg.transforms import HorizontalFlip


class TestFlipTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_horizontal_flip(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = HorizontalFlip(location_dict=get_locations(
            DEAPDataset.channels, DEAPDataset.grid), channel_dim=0)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 128))

    def test_repr_uses_current_configuration(self):
        self.assertIn('location_dict=', repr(
            HorizontalFlip({'L': (0, 0), 'R': (0, 2)})))


if __name__ == '__main__':
    unittest.main()
