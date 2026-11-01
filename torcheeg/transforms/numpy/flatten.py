from typing import Dict, Union

import numpy as np

from ..base_transform import EEGTransform


class Flatten(EEGTransform):
    r"""
    Flatten the input EEG representation.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Flatten()
        t(np.random.randn(62, 5)).shape
        >>> (310,)

    .. automethod:: __call__
    """

    def __call__(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg.reshape(-1)
