from typing import Union, Dict, List

import numpy as np

from ..base_transform import EEGTransform
from scipy.signal import resample


class Downsample(EEGTransform):
    r'''
    Downsample the EEG signal to a specified number of data points.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Downsample(num_points=32, axis=-1)
        # normalize along the first dimension (electrode dimension)
        t(eeg=np.random.randn(32, 128))['eeg'].shape
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
    




class SetSamplingRate(EEGTransform):
    r'''
    Change the EEG signal to another sampling rate.

    .. code-block:: python

        from torcheeg import transforms

        t = SetSamplingRate(origin=500,target_sampling_rate=128)
        t(eeg=np.random.randn(32, 1000))['eeg'].shape
        >>> (32, 256)

    Args:
        origin (int): Original sampling rate of EEG.
        target_sampling_rate (int): Target sampling rate of EEG.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,origin:int, target_sampling_rate:int, apply_to_baseline=False):
        super(SetSamplingRate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.original_rate = origin
        self.new_rate = target_sampling_rate


    def apply(self, eeg, **kwargs) -> any:
        new_length = int(eeg.shape[-1] *  self.new_rate/ self.original_rate)
    
        result = []
        
        eeg_ = eeg.reshape(-1,eeg.shape[-1])

        for signal in eeg_:
            resampled_signal = resample(signal, new_length)
          
            result.append(resampled_signal)
        result = np.stack(result,axis=0)
        return result.reshape(*eeg.shape[:-1],new_length)
        
    
    @property
    def __repr__(self)->any :
        return  f'''{
                'original_sampling_rate': self.original_rate,
                'target_sampling_rate': self.new_rate,
                'apply_to_baseline':self.apply_to_baseline
            }'''
