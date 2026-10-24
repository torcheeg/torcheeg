from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from pylab import cm

from .utils import _plot_to_image


def plot_3d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 3-D matrices in 3-D space.

    .. code-block:: python

        from torcheeg.visualizers import plot_3d_tensor

        eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(eeg)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_3d_tensor.png
        :alt: The output image of plot_3d_tensor
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input 3-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)

    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.detach().cpu().numpy()

    filled = np.ones_like(ndarray, dtype=bool)
    colormap = cm.get_cmap(color)

    ndarray = (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(filled,
              facecolors=colormap(ndarray),
              edgecolors='k',
              linewidths=0.1,
              shade=False)

    img = _plot_to_image(fig)
    plt.show()

    return np.array(img)



def plot_2d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 2-D matrices in 2-D space.

    .. code-block:: python

        import torch

        from torcheeg.visualizers import plot_2d_tensor

        eeg = torch.randn(9, 9)
        img = plot_2d_tensor(eeg)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_2d_tensor.png
        :alt: The output image of plot_2d_tensor
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input 2-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)

    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.detach().cpu().numpy()

    fig = plt.figure()
    ax = plt.axes()

    colormap = cm.get_cmap(color)
    ax.imshow(ndarray, cmap=colormap, interpolation='nearest')

    img = _plot_to_image(fig)
    plt.show()

    return np.array(img)

