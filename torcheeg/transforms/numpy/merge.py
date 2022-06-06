import numpy as np

from typing import Callable, Sequence


class Concatenate:
    r'''
    Merge the calculation results of multiple transforms, which are used when feature fusion is required.

    .. code-block:: python

        transform = Concatenate([
            BandDifferentialEntropy(),
            BandMeanAbsoluteDeviation()
        ])
        transform(torch.randn(32, 128)).shape
        >>> (32, 8)

    Args:
        transforms (list, tuple): a sequence of transforms
    
    .. automethod:: __call__
    '''
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            
        Returns:
            np.ndarray: The combined results of multiple transforms.
        '''
        out = []
        for t in self.transforms:
            out.append(t(eeg))
        return np.concatenate(out, axis=-1)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string