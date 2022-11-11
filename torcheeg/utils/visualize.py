import io
import itertools
import random
from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import PIL
import torch
from matplotlib import colors
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from pylab import cm

mne.set_log_level('CRITICAL')

default_montage = mne.channels.make_standard_montage('standard_1020')


def plot2image(ploter):
    buf = io.BytesIO()
    ploter.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    return PIL.Image.open(buf)


def plot_raw_topomap(tensor: torch.Tensor,
                     channel_list: List[str],
                     sampling_rate: int,
                     plot_second_list: List[int] = [0.0, 0.25, 0.5, 0.75],
                     montage: mne.channels.DigMontage = default_montage):
    r'''
    Plot a topographic map of the input raw EEG signal as image.

    .. code-block:: python

        eeg = torch.randn(32, 128)
        img = plot_raw_topomap(eeg,
                         channel_list=DEAP_CHANNEL_LIST,
                         sampling_rate=128)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_raw_topomap.png
        :alt: The output image of plot_raw_topomap
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, number of data points].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        sampling_rate (int): Sample rate of the data.
        plot_second_list (list): The time (second) at which the topographic map is drawn. (default: :obj:`[0.0, 0.25, 0.5, 0.75]`)
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['eeg'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list,
                           ch_types=ch_types,
                           sfreq=sampling_rate)
    tensor = tensor.detach().cpu().numpy()
    info.set_montage(montage,
                     match_alias=True,
                     match_case=False,
                     on_missing='ignore')
    fig, axes = plt.subplots(1, len(plot_second_list), figsize=(20, 5))
    for i, label in enumerate(plot_second_list):
        mne.viz.plot_topomap(tensor[:, int(sampling_rate * label)],
                             info,
                             axes=axes[i],
                             show=False,
                             sphere=(0., 0., 0., 0.11))
        axes[i].set_title(f'{label}s', {
            'fontsize': 24,
            'fontname': 'Liberation Serif'
        })

    img = plot2image(fig)
    plt.show()
    return np.array(img)


def plot_feature_topomap(tensor: torch.Tensor,
                         channel_list: List[str],
                         feature_list: Union[List[str], None] = None,
                         montage: mne.channels.DigMontage = default_montage):
    r'''
    Plot a topographic map of the input EEG features as image.

    .. code-block:: python

        eeg = torch.randn(32, 4)
        img = plot_feature_topomap(eeg,
                         channel_list=DEAP_CHANNEL_LIST,
                         feature_list=["theta", "alpha", "beta", "gamma"])
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_feature_topomap.png
        :alt: The output image of plot_feature_topomap
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, dimensions of features].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        feature_list (list): . The names of feature dimensions displayed on the output image, whose length should be consistent with the dimensions of features. If set to None, the dimension index of the feature is used instead. (default: :obj:`None`)
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['eeg'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list, ch_types=ch_types, sfreq=128)
    tensor = tensor.detach().cpu().numpy()
    info.set_montage(montage,
                     match_alias=True,
                     match_case=False,
                     on_missing='ignore')

    if feature_list is None:
        feature_list = list(range(tensor.shape[1]))
    num_subplots = len(feature_list)

    fig, axes = plt.subplots(1, num_subplots, figsize=(num_subplots * 5, 5))

    if num_subplots > 1:
        for i, (label) in enumerate(feature_list):
            mne.viz.plot_topomap(tensor[:, i],
                                 info,
                                 axes=axes[i],
                                 show=False,
                                 sphere=(0., 0., 0., 0.11))
            axes[i].set_title(label, {
                'fontsize': 24,
                'fontname': 'Liberation Serif'
            })
    else:
        mne.viz.plot_topomap(tensor[:, 0],
                             info,
                             axes=axes,
                             show=False,
                             sphere=(0., 0., 0., 0.11))
        axes.set_title(feature_list[0], {
            'fontsize': 24,
            'fontname': 'Liberation Serif'
        })

    img = plot2image(fig)
    plt.show()
    return np.array(img)


def plot_signal(tensor: torch.Tensor,
                channel_list: List[str],
                sampling_rate: int,
                montage: mne.channels.DigMontage = default_montage):
    r'''
    Plot signal values of the input raw EEG as image.

    .. code-block:: python

        eeg = torch.randn(32, 128)
        img = plot_signal(eeg,
                          channel_list=DEAP_CHANNEL_LIST,
                          sampling_rate=128)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_signal.png
        :alt: The output image of plot_signal
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, number of data points].
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        sampling_rate (int): Sample rate of the data.
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ch_types = ['misc'] * len(channel_list)
    info = mne.create_info(ch_names=channel_list,
                           ch_types=ch_types,
                           sfreq=sampling_rate)

    epochs = mne.io.RawArray(tensor.detach().cpu().numpy(), info)
    epochs.set_montage(montage, match_alias=True, on_missing='ignore')
    img = plot2image(
        epochs.plot(show_scrollbars=False, show_scalebars=False, block=True))
    plt.show()
    return np.array(img)


def plot_3d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 3-D matrices in 3-D space.

    .. code-block:: python

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

    img = plot2image(fig)
    plt.show()

    return np.array(img)


def plot_2d_tensor(tensor: torch.Tensor,
                   color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 2-D matrices in 2-D space.

    .. code-block:: python

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

    img = plot2image(fig)
    plt.show()

    return np.array(img)


def plot_adj_connectivity(adj: torch.Tensor,
                          channel_list: list = None,
                          region_list: list = None,
                          num_connectivity: int = 60,
                          linewidth: float = 1.5):
    r'''
    Visualize connectivity between nodes in an adjacency matrix, using circular networks.

    .. code-block:: python

        adj = torch.randn(62, 62) # relationship between 62 electrodes
        img = plot_adj_connectivity(adj, SEED_CHANNEL_LIST)
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
    assert len(channel_list) == adj.shape[0] and len(channel_list) == adj.shape[
        1], 'The size of the adjacency matrix does not match the number of channel names.'

    node_colors = None
    if region_list:
        num_region = len(region_list)
        colormap = matplotlib.cm.get_cmap('rainbow')
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

    node_angles = circular_layout(channel_list,
                                  channel_list,
                                  start_pos=90)
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
    img = plot2image(fig)
    plt.show()

    return np.array(img)
