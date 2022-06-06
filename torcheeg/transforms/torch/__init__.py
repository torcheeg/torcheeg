import torch
import numpy as np

from .augment import *

class ToTensor:
    r'''
    Convert a :obj:`numpy.ndarray` to tensor. Different from :obj:`torchvision`, tensors are returned without scaling.

    .. code-block:: python

        transform = ToTensor()
        transform(np.random.randn(32, 128)).shape
        >>> (32, 128)

    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        r'''
        Args:
            x (np.ndarray): The input EEG signals.

        Returns:
            torch.Tensor: The output represented by :obj:`torch.Tensor`.
        '''
        return torch.from_numpy(x).float()

    def __repr__(self):
        return f"{self.__class__.__name__}()"