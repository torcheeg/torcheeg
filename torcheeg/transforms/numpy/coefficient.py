from typing import Union, Dict

import numpy as np
from spectrum import aryule

from ..base_transform import EEGTransform


class ARRCoefficient(EEGTransform):
    r'''
    Calculate autoregression reflection coefficients on the input data.

    .. code-block:: python

        transform = ARRCoefficient(order=4)
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        order (int): The order of autoregressive process to be fitted. (default: :obj:`4`)
        norm (str): Use a biased or unbiased correlation. (default: :obj:`biased`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 order: int = 4,
                 norm: str = 'biased',
                 apply_to_baseline: bool = False):
        super(ARRCoefficient,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.order = order
        self.norm = norm

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals or features.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray [number of electrodes, order]: The autoregression reflection coefficients.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

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