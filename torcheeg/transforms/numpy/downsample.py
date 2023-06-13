from typing import Union, Dict, List

import numpy as np

from ..base_transform import EEGTransform


class Downsample(EEGTransform):
    r'''
    Downsample the EEG signal to a specified number of data points.

    .. code-block:: python

        transform = Downsample(num_points=32, axis=-1)
        # normalize along the first dimension (electrode dimension)
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 32)

    Args:
        num_points (int): The number of data points after downsampling.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized. (default: :obj:`-1`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 num_points: int,
                 axis: Union[int, None] = -1,
                 apply_to_baseline: bool = False):
        super(Downsample, self).__init__(apply_to_baseline=apply_to_baseline)
        self.num_points = num_points
        self.axis = axis

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
            np.ndarray: The normalized results.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs):
        times_tamps = np.linspace(0,
                                  eeg.shape[self.axis] - 1,
                                  self.num_points,
                                  dtype=int)
        return eeg.take(times_tamps, axis=self.axis)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'num_points': self.num_points,
            'axis': self.axis
        })