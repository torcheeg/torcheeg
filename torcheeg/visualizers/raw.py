from typing import List

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from .utils import _plot_to_image

_DEFAULT_MONTAGE = mne.channels.make_standard_montage('standard_1020')


def plot_signal(tensor: torch.Tensor,
                channel_list: List[str],
                sampling_rate: int,
                montage: mne.channels.DigMontage = _DEFAULT_MONTAGE):
    r'''
    Plot signal values of the input raw EEG as image.

    .. code-block:: python

        import torch

        from torcheeg.visualizers import plot_signal
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

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
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.datasets.constants` module.
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
    img = _plot_to_image(
        epochs.plot(show_scrollbars=False, show_scalebars=False, block=True))
    plt.show()
    return np.array(img)

