from typing import Dict, Union

import numpy as np
import torch

from ..base_transform import EEGTransform


class ToTensor(EEGTransform):
    r"""
    Convert a :obj:`numpy.ndarray` to tensor. Different from :obj:`torchvision`, tensors are returned without scaling.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.ToTensor()
        t(np.random.randn(32, 128)).shape
        >>> (32, 128)

    Args:


    .. automethod:: __call__
    """

    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, eeg: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(eeg).float()
