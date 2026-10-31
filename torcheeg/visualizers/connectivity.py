import matplotlib.pyplot as plt
import numpy as np
import torch
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle

from .utils import _plot_to_image


def plot_adj_connectivity(adj: torch.Tensor,
                          channel_list: list = None,
                          region_list: list = None,
                          num_connectivity: int = 60,
                          linewidth: float = 1.5):
    r'''
    Visualize connectivity between nodes in an adjacency matrix, using circular networks.

    .. code-block:: python

        import torch

        from torcheeg.visualizers import plot_adj_connectivity
        from torcheeg.datasets import SEEDDataset

        adj = torch.randn(62, 62) # relationship between 62 electrodes
        img = plot_adj_connectivity(adj, SEEDDataset.channels)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_adj_connectivity.png
        :alt: The output image of plot_adj_connectivity
        :align: center

    |

    Args:
        adj (torch.Tensor): The input 2-D adjacency tensor.
        channel_list (list): The electrode name of the row/column in the input adjacency matrix, used to label the electrode corresponding to the node on circular networks. If set to None, the electrode's index is used. (default: :obj:`None`)
        region_list (list): region_list (list): The region list where the electrodes are divided into different brain regions. If set, electrodes in the same area will be aligned on the map and filled with the same color. (default: :obj:`None`)
        num_connectivity (int): The number of connections to retain on circular networks, where edges with larger weights in the adjacency matrix will be limitedly retained, and the excess is omitted. (default: :obj:`50`)
        linewidth (float): Line width to use for connections. (default: :obj:`1.5`)

    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    if channel_list is None:
        channel_list = list(range(len(adj)))
    adj = adj.detach().cpu().numpy()
    if not (len(channel_list) == adj.shape[0] and len(channel_list) == adj.shape[1]):
        raise ValueError(
            'The size of the adjacency matrix does not match the number of channel names.'
        )

    node_colors = None
    if region_list:
        num_region = len(region_list)
        colormap = plt.get_cmap('rainbow')
        region_colors = list(colormap(np.linspace(0, 1, num_region)))
        # random.shuffle(region_colors)
        # # circle two colors is better
        # colors = colormap(np.linspace(0, 1, 4))
        # region_colors = [
        #     colors[1] if region_index % 2 == 0 else colors[2]
        #     for region_index in range(num_region)
        # ]

        new_channel_list = []
        new_adj_order = []
        for region_index, region in enumerate(region_list):
            for electrode_index in region:
                new_adj_order.append(electrode_index)
                new_channel_list.append(channel_list[electrode_index])
        new_adj = adj[new_adj_order][:, new_adj_order]

        electrode_colors = [None] * len(new_channel_list)
        i = 0
        for region_index, region in enumerate(region_list):
            for electrode_index in region:
                electrode_colors[i] = region_colors[region_index]
                i += 1

        adj = new_adj
        channel_list = new_channel_list
        node_colors = electrode_colors

    node_angles = circular_layout(channel_list, channel_list, start_pos=90)
    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax = plt.subplots(figsize=(8, 8),
                           facecolor='white',
                           subplot_kw=dict(polar=True))
    plot_connectivity_circle(adj,
                             channel_list,
                             node_colors=node_colors,
                             n_lines=num_connectivity,
                             node_angles=node_angles,
                             ax=ax,
                             facecolor='white',
                             textcolor='black',
                             node_edgecolor='white',
                             colormap='autumn',
                             colorbar=False,
                             padding=0.0,
                             linewidth=linewidth,
                             fontsize_names=16)
    fig.tight_layout()
    img = _plot_to_image(fig)
    plt.show()

    return np.array(img)
