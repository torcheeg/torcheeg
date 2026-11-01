from typing import Dict, Union

import numpy as np
from spectrum import aryule

from ..base_transform import EEGTransform


class ARRCoefficient(EEGTransform):
    r"""
    Calculate autoregression reflection coefficients on the input data.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.ARRCoefficient(order=4)
        t(np.random.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        order (int): The order of autoregressive process to be fitted. (default: :obj:`4`)
        norm (str): Use a biased or unbiased correlation. (default: :obj:`biased`)


    .. automethod:: __call__
    """

    def __init__(self, order: int = 4, norm: str = 'biased'):
        super(ARRCoefficient, self).__init__()
        self.order = order
        self.norm = norm

    def apply(self, eeg: np.ndarray, **kwargs):
        c_list = []
        for c in eeg:
            ar_coeffs, _, _ = aryule(c, order=self.order, norm=self.norm)
            c_list.append(ar_coeffs)
        return np.array(c_list)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'order': self.order,
            'norm': self.norm
        })
