from typing import Dict, Union, Tuple

import numpy as np

from ..base_transform import EEGTransform


class Reshape(EEGTransform):
    r'''
    Reshape the input EEG representation to the specified shape.

    .. code-block:: python

        from torcheeg import transforms
        
        t = transforms.Reshape((6, 15, 4))
        t(eeg=np.random.randn(6, 60))['eeg'].shape
        >>> (6, 15, 4)

    .. automethod:: __call__
    '''
    def __init__(self, shape: Tuple[int, ...]) -> None:
        r'''
        Args:
            shape (Tuple[int, ...]): The target shape to reshape the input EEG signals to.
        '''
        super().__init__()
        self.shape = shape
        
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
        return eeg.reshape(self.shape)