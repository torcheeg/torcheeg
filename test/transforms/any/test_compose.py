import unittest

import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.utils import get_locations
from torcheeg.transforms import (BandDifferentialEntropy, Binary, Compose,
                                 MeanStdNormalize, RandomMask, RandomNoise,
                                 Resize, Select, ToGrid, ToTensor)


class TestComposeTransforms(unittest.TestCase):

    def test_compose(self):
        # torch
        transform = Compose([
            ToTensor(),
            Resize(size=(64, 64)),
            RandomNoise(p=0.1),
            RandomMask(p=0.1)
        ])

        self.assertEqual(
            tuple(transform(eeg=np.random.randn(128, 9, 9))['eeg'].shape),
            (128, 64, 64))

        # numpy
        transform = Compose([
            BandDifferentialEntropy(),
            MeanStdNormalize(),
            ToGrid(get_locations(DEAPDataset.channels, DEAPDataset.grid))
        ])
        self.assertEqual(
            tuple(transform(eeg=np.random.randn(32, 128))['eeg'].shape),
            (4, 9, 9))

        # label
        info = {'valence': 4.5, 'arousal': 5.5, 'subject_id': 7}
        transform = Compose([Select(key='valence'), Binary(threshold=5.0)])
        self.assertEqual(transform(y=info)['y'], 0)


if __name__ == '__main__':
    unittest.main()
