import io
import itertools
from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
import PIL
import torch
import torch_geometric
from matplotlib import colors
from pylab import cm
from torch_geometric.data import Data

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
        :width: 400px
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
    info = mne.create_info(ch_names=channel_list, ch_types=ch_types, sfreq=sampling_rate)
    tensor = tensor.detach().cpu().numpy()
    info.set_montage(montage, match_alias=True, match_case=False, on_missing='ignore')
    fig, axes = plt.subplots(1, len(plot_second_list), figsize=(20, 5))
    for i, label in enumerate(plot_second_list):
        mne.viz.plot_topomap(tensor[:, int(sampling_rate * label)],
                             info,
                             axes=axes[i],
                             show=False,
                             sphere=(0., 0., 0., 0.11))
        axes[i].set_title(f'{label}s', {'fontsize': 24, 'fontname': 'Liberation Serif'})

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
        :width: 400px
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
    info.set_montage(montage, match_alias=True, match_case=False, on_missing='ignore')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    if feature_list is None:
        feature_list = list(range(tensor.shape[1]))

    for i, (label) in enumerate(feature_list):
        mne.viz.plot_topomap(tensor[:, i], info, axes=axes[i], show=False, sphere=(0., 0., 0., 0.11))
        axes[i].set_title(label, {'fontsize': 24, 'fontname': 'Liberation Serif'})

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
        :width: 400px
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
    info = mne.create_info(ch_names=channel_list, ch_types=ch_types, sfreq=sampling_rate)

    epochs = mne.io.RawArray(tensor.detach().cpu().numpy(), info)
    epochs.set_montage(montage, match_alias=True, on_missing='ignore')
    img = plot2image(epochs.plot(show_scrollbars=False, show_scalebars=False, block=True))
    plt.show()
    return np.array(img)


def plot_3d_tensor(tensor: torch.Tensor, color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 3-D matrices in 3-D space.

    .. code-block:: python

        eeg = torch.randn(128, 9, 9)
        img = plot_3d_tensor(eeg)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_3d_tensor.png
        :width: 200px
        :alt: The output image of plot_3d_tensor
        :align: center

    |

    Args:
        tensor (torch.Tensor): The input 3-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.numpy()

    filled = np.ones_like(ndarray, dtype=bool)
    colormap = cm.get_cmap(color)

    ndarray = (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(filled, facecolors=colormap(ndarray), edgecolors='k', linewidths=0.1, shade=False)

    img = plot2image(fig)
    plt.show()

    return np.array(img)


def plot_2d_tensor(tensor: torch.Tensor, color: Union[colors.Colormap, str] = 'hsv'):
    r'''
    Visualize a 2-D matrices in 2-D space.

    .. code-block:: python

        eeg = torch.randn(9, 9)
        img = plot_2d_tensor(eeg)
        # If using jupyter, the output image will be drawn on notebooks.

    .. image:: _static/plot_2d_tensor.png
        :width: 200px
        :alt: The output image of plot_2d_tensor
        :align: center

    |
    
    Args:
        tensor (torch.Tensor): The input 2-D tensor.
        color (colors.Colormap or str): The color map used for the face color of the axes. (default: :obj:`hsv`)
    
    Returns:
        np.ndarray: The output image in the form of :obj:`np.ndarray`.
    '''
    ndarray = tensor.numpy()

    fig = plt.figure()
    ax = plt.axes()

    colormap = cm.get_cmap(color)
    ax.imshow(ndarray, cmap=colormap, interpolation='nearest')

    img = plot2image(fig)
    plt.show()

    return np.array(img)


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
        :width: 300px
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
    edge_weights = [edgedata["edge_weight"] for _, _, edgedata in g.edges(data=True)]
    colormap = cm.get_cmap(color)
    edge_colors = colormap(edge_weights)

    # get posistion of nodes
    # flip bottom down
    max_pos = max(list(itertools.chain(*channel_location_dict.values())))
    # rot 90
    pos = {i: [v[1], max_pos - v[0]] for i, v in enumerate(channel_location_dict.values())}
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
