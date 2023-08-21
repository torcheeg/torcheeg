from typing import Dict, Union

import numpy as np

from ..base_transform import EEGTransform


class Flatten(EEGTransform):
    r'''
    Flatten the input EEG representation.

    .. code-block:: python

        transform = Flatten()
        transform(eeg=np.random.randn(62, 5))['eeg'].shape
        >>> (310,)

    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals.
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The transformed results.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg.reshape(-1)
