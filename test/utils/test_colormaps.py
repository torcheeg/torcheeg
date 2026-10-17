import unittest
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torcheeg.datasets.constants import (DEAP_ADJACENCY_MATRIX,
                                         DEAP_CHANNEL_LOCATION_DICT)
from torcheeg.transforms.graph import ToG
from torcheeg.utils import (plot_2d_tensor, plot_3d_tensor,
                            plot_adj_connectivity, plot_graph)


class TestColormaps(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_named_and_object_colormaps(self):
        graph = ToG(DEAP_ADJACENCY_MATRIX)(
            eeg=np.arange(32 * 8).reshape(32, 8))['eeg']
        workflows = (
            lambda color: plot_2d_tensor(torch.arange(6).reshape(2, 3),
                                         color=color),
            lambda color: plot_3d_tensor(torch.arange(8).reshape(2, 2, 2),
                                         color=color),
            lambda color: plot_graph(graph, DEAP_CHANNEL_LOCATION_DICT,
                                     color=color),
        )
        for index, workflow in enumerate(workflows):
            with self.subTest(workflow=index), warnings.catch_warnings():
                warnings.simplefilter('error', matplotlib.MatplotlibDeprecationWarning)
                named = workflow('viridis')
                plt.close('all')
                custom = workflow(plt.get_cmap('viridis'))
                self.assertEqual(named.ndim, 3)
                self.assertEqual(named.dtype, np.uint8)
                self.assertGreater(np.ptp(named[..., :3]), 0)
                np.testing.assert_array_equal(named, custom)
                plt.close('all')

    def test_connectivity_region_colors(self):
        adjacency = torch.tensor([[0., 0.5, 0.2],
                                  [0.5, 0., 0.8],
                                  [0.2, 0.8, 0.]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', matplotlib.MatplotlibDeprecationWarning)
            image = plot_adj_connectivity(adjacency, ['A', 'B', 'C'],
                                           [[0, 1], [2]], num_connectivity=3)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(np.ptp(image[..., :3]), 0)


if __name__ == '__main__':
    unittest.main()
