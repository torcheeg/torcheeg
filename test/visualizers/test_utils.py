import unittest

import matplotlib.pyplot as plt
import numpy as np

from torcheeg.visualizers.utils import _plot_to_image


class TestPlotToImage(unittest.TestCase):

    def test_figure_is_converted_to_a_readable_image(self):
        figure, axes = plt.subplots(figsize=(2, 1))
        self.addCleanup(plt.close, figure)
        axes.plot([0, 1], [0, 1], color='red')
        image = _plot_to_image(figure)
        self.addCleanup(image.close)
        image.load()
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)
        pixels = np.asarray(image)
        self.assertGreater(np.ptp(pixels[..., :3]), 0)


if __name__ == '__main__':
    unittest.main()
