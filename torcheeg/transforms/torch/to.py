from typing import Dict, Union

import numpy as np

import torch

from ..base_transform import EEGTransform


class ToTensor(EEGTransform):
    r'''
    Convert a :obj:`numpy.ndarray` to tensor. Different from :obj:`torchvision`, tensors are returned without scaling.

    .. code-block:: python

        transform = ToTensor()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            dict: If baseline is passed and apply_to_baseline is set to True, then {'eeg': ..., 'baseline': ...}, else {'eeg': ...}. The output is represented by :obj:`torch.Tensor`.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        return torch.from_numpy(eeg).float()