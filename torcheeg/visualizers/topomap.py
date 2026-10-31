from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from .utils import _plot_to_image

_DEFAULT_MONTAGE = mne.channels.make_standard_montage('standard_1020')


def plot_raw_topomap(tensor: torch.Tensor,
                     channel_list: List[str],
                     sampling_rate: int,
                     plot_second_list: List[int] = [0.0, 0.25, 0.5, 0.75],
                     montage: mne.channels.DigMontage = _DEFAULT_MONTAGE):
    r'''
    Plot a topographic map of the input raw EEG signal as image.

    .. code-block:: python

        from torcheeg.visualizers import plot_raw_topomap
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

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
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.datasets.constants` module.
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

    img = _plot_to_image(fig)
    plt.show()
    return np.array(img)


def plot_feature_topomap(tensor: torch.Tensor,
                         channel_list: List[str],
                         feature_list: Union[List[str], None] = None,
                         montage: mne.channels.DigMontage = _DEFAULT_MONTAGE,
                         fig_shape: Tuple[int, int] = None):
    r'''
    Plot a topographic map of the input EEG features as image.

    .. code-block:: python

        from torcheeg.visualizers import plot_feature_topomap
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

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
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.datasets.constants` module.
        feature_list (list): . The names of feature dimensions displayed on the output image, whose length should be consistent with the dimensions of features. If set to None, the dimension index of the feature is used instead. (default: :obj:`None`)
        montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
        fig_shape (Tuple[int, int], optional): The shape of the sub graphs (width, height). If `None`, the layout is automatically set to (1, len(feature_list)). (default: :obj:`None`)

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

    if fig_shape == None:
        fig_shape = (1, num_subplots)
    else:
        if len(fig_shape) != 2:
            raise ValueError(
                "fig_shape only support 2d graph, so just contain width and height"
            )
        if not all(isinstance(n, int) and n > 0 for n in fig_shape):
            raise ValueError(
                "width and height in fig_shape must be positive integers")
        if fig_shape[0] * fig_shape[1] != num_subplots:
            raise ValueError(
                f"The product of width and height in fig_shape must equal feature_list length: {num_subplots}"
            )

    fig, axes = plt.subplots(fig_shape[0],
                             fig_shape[1],
                             figsize=(fig_shape[1] * 5, fig_shape[0] * 5),
                             squeeze=False)

    if num_subplots > 1:
        for i, (label) in enumerate(feature_list):
            row, col = i // fig_shape[1], i % fig_shape[1]
            mne.viz.plot_topomap(tensor[:, i],
                                 info,
                                 axes=axes[row, col],
                                 show=False,
                                 sphere=(0., 0., 0., 0.11))
            axes[row, col].set_title(label, {
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

    img = _plot_to_image(fig)
    plt.show()
    return np.array(img)
