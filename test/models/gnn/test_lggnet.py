import unittest

import torch

from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.montage import GENERAL_REGION_LIST
from torcheeg.datasets.utils import get_regions
from torcheeg.models import LGGNet


class TestLGGNet(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.rand(2, 1, 32, 128)
        model = LGGNet(get_regions(DEAPDataset.channels, GENERAL_REGION_LIST),
                       num_electrodes=32, chunk_size=128)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
