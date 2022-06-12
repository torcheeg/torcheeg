from typing import Callable, Sequence, Dict, Union

import numpy as np

from ..base_transform import EEGTransform


class Concatenate(EEGTransform):
    r'''
    Merge the calculation results of multiple transforms, which are used when feature fusion is required.

    .. code-block:: python

        transform = Concatenate([
            BandDifferentialEntropy(),
            BandMeanAbsoluteDeviation()
        ])
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 8)

    Args:
        transforms (list, tuple): a sequence of transforms
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transforms: Sequence[Callable],
                 apply_to_baseline: bool = False):
        super(Concatenate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.transforms = transforms

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            np.ndarray: The combined results of multiple transforms.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        out = []
        for t in self.transforms:
            out.append(t.apply(eeg, **kwargs))
        return np.concatenate(out, axis=-1)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
