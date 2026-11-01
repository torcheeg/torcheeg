from typing import Dict, Tuple, Union

import numpy as np

from ..base_transform import EEGTransform


class Reshape(EEGTransform):
    r"""
    Reshape the input EEG representation to the specified shape.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Reshape((6, 15, 4))
        t(np.random.randn(6, 60)).shape
        >>> (6, 15, 4)

    .. automethod:: __call__
    """

    def __init__(self, shape: Tuple[int, ...]) -> None:
        r'''
        Args:
            shape (Tuple[int, ...]): The target shape to reshape the input EEG signals to.
        '''
        super().__init__()
        self.shape = shape

    def __call__(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg.reshape(self.shape)
