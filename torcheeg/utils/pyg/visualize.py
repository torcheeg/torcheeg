import io
import itertools
from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL
import torch_geometric
from matplotlib import colors
from pylab import cm
from torch_geometric.data import Data


def plot2image(ploter):
    buf = io.BytesIO()
    ploter.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    return PIL.Image.open(buf)


def plot_graph(data: Data,
               channel_location_dict: Dict[str, List[int]],
               color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a graph structure. For the electrode position information, please refer to constants grouped by dataset:

    - datasets.constants.emotion_recognition.deap.DEAP_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.dreamer.DREAMER_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.seed.SEED_CHANNEL_LOCATION_DICT
    - ...

    .. code-block:: python

        eeg = np.random.randn(32, 128)
        g = ToG(DEAP_ADJACENCY_MATRIX)(eeg=eeg)['eeg']
        img = plot_graph(g)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_graph.png
        :alt: The output image of plot_graph
        :align: center

    |
    
    Args:
        data (torch_geometric.data.Data): The input graph structure represented by torch_geometric.
        channel_location_dict (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to the row index and column index of the electrode on the grid.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    fig = plt.figure()

    # convert to networkx
    edge_attrs = ['edge_weight'] * len(data.edge_weight.tolist())
    g = torch_geometric.utils.to_networkx(data, edge_attrs=edge_attrs)

    # get color of edges
    edge_weights = [
        edgedata["edge_weight"] for _, _, edgedata in g.edges(data=True)
    ]
    colormap = cm.get_cmap(color)
    edge_colors = colormap(edge_weights)

    # get posistion of nodes
    # flip bottom down
    max_pos = max(list(itertools.chain(*channel_location_dict.values())))
    # rot 90
    pos = {
        i: [v[1], max_pos - v[0]]
        for i, v in enumerate(channel_location_dict.values())
    }
    labels = {i: v for i, v in enumerate(channel_location_dict.keys())}

    # draw network
    nx.draw_networkx(g,
                     node_size=550,
                     node_color='w',
                     edgecolors='w',
                     pos=pos,
                     labels=labels,
                     with_labels=True,
                     edge_color=edge_colors)

    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=colormap))

    # remove margin
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    img = plot2image(fig)
    plt.show()

    return np.array(img)
