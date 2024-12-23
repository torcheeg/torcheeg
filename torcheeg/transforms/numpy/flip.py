from copy import deepcopy
from typing import Dict, Union

import numpy as np

from torcheeg.transforms import EEGTransform


def get_swap_pair(location_dict):
    location_values = list(location_dict.values())
    eeg_width = np.array(location_values)[:, 1].max()
    visited = [False for _ in location_values]
    swap_pair = []
    for i, loc in enumerate(location_values):
        if visited[i]:
            continue
        x, y = loc
        target_loc = [x, eeg_width - y]
        for j, loc_j in enumerate(location_values[i:]):
            # print(loc_j)
            if target_loc == loc_j:
                swap_pair.append((i, j+i))
                visited[i] = True
                visited[j] = True
                break
    return swap_pair


def horizontal_flip(eeg, eeg_channel_dim, pair):
    eeg = deepcopy(eeg)

    for i, (index1, index2) in enumerate(pair):
        slice_tuple1 = tuple(
            slice(None) if i != eeg_channel_dim else index1 for i in range(eeg.ndim))
        slice_tuple2 = tuple(
            slice(None) if i != eeg_channel_dim else index2 for i in range(eeg.ndim))
        t = deepcopy(eeg[slice_tuple1])
        eeg[slice_tuple1] = eeg[slice_tuple2]
        eeg[slice_tuple2] = t
    return eeg


class HorizontalFlip(EEGTransform):
    r'''
    Flip the EEG signal horizontally based on the electrode's position.

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets.constants.motor_imagery import BCICIV2A_LOCATION_DICT

        eeg = np.random.randn(32, 4, 22, 128)
        t = transforms.HorizontalFlip(
            location_dict=BCICIV2A_LOCATION_DICT,
            channel_dim=2
        )
        t(eeg=eeg)['eeg'].shape
        >>> (32, 4, 22, 128)

    Args:
        location_dict (dict): The dict of electrodes and their postions. 
        channel_dim (int): The dim of electrodes in EEG data.

    .. automethod:: __call__
    '''

    def __init__(self,
                 location_dict: Union[dict, None],
                 channel_dim: int = 0):

        super().__init__(apply_to_baseline=False)
        self.location_dict = location_dict
        self.swap_pair = get_swap_pair(location_dict)
        self.channel_dim = channel_dim

    def apply(self, eeg: any, **kwargs) -> any:
        eeg = horizontal_flip(eeg, self.channel_dim, self.swap_pair)
        return eeg

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'apply_to_baseline': self.apply_to_baseline,
                                          'loaction_dict': self.location_dict,
                                          'channel_dim': self.channel_dim})
