import unittest

import torch

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.montage import GENERAL_REGION_LIST
from torcheeg.datasets.utils import get_regions
from torcheeg.models import LGGNet


class TestLggnet(unittest.TestCase):

    def test_lggnet(self):
        eeg = torch.rand(2, 1, 32, 128)
        model = LGGNet(get_regions(DEAPDataset.channels, GENERAL_REGION_LIST),
                       num_electrodes=32, chunk_size=128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
