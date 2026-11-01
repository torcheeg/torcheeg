import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from torcheeg.datasets import SEEDDataset
from torcheeg.datasets.montage import GENERAL_REGION_LIST
from torcheeg.datasets.utils import get_regions
from torcheeg.visualizers import plot_adj_connectivity


class TestConnectivity(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

    def test_plot_adj_connectivity(self):
        mock_adj = torch.randn(62, 62)
        img = plot_adj_connectivity(mock_adj,
                                    SEEDDataset.channels,
                                    get_regions(SEEDDataset.channels,
                                                GENERAL_REGION_LIST),
                                    num_connectivity=60,
                                    linewidth=1.5)

        self.assertFalse(np.all(img == 0))


if __name__ == '__main__':
    unittest.main()
