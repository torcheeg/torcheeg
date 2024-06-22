from typing import Union, Dict, List

import numpy as np

from ..base_transform import EEGTransform
import scipy


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
    Resample a EEG series from orig_sr to target_sr.

    .. code-block:: python

        from torcheeg import transforms

        t = SetSamplingRate(origin=500,target_sampling_rate=128)
        t(eeg=np.random.randn(32, 1000))['eeg'].shape
        >>> (32, 256)

    Args:
        origin (int): Original sampling rate of EEG.
        target_sampling_rate (int): Target sampling rate of EEG.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized. (default: :obj:`-1`)
        scale (bool, optional): Whether to scale the resampled signal so that ``y`` and ``y_hat`` have approximately equal total energy. (default: :obj:`False`)
        res_type (str, optional): The resampling method to use. (default: :obj:`'soxr_hq'`), options:

            'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
                `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
                ``'soxr_hq'`` is the default setting of `soxr`.
            'soxr_qq'
                `soxr` Quick cubic interpolation (very fast, but not bandlimited)
            'kaiser_best'
                `resampy` high-quality mode
            'kaiser_fast'
                `resampy` faster method
            'fft' or 'scipy'
                `scipy.signal.resample` Fourier method.
            'polyphase'
                `scipy.signal.resample_poly` polyphase filtering. (fast)
            'linear'
                `samplerate` linear interpolation. (very fast, but not bandlimited)
            'zero_order_hold'
                `samplerate` repeat the last value between samples. (very fast, but not bandlimited)
            'sinc_best', 'sinc_medium' or 'sinc_fastest'
                `samplerate` high-, medium-, and low-quality bandlimited sinc interpolation.

            .. note::
                Not all options yield a bandlimited interpolator. If you use `soxr_qq`, `polyphase`,
                `linear`, or `zero_order_hold`, you need to be aware of possible aliasing effects.

            .. note::
                `samplerate` and `resampy` are not installed with `torcheeg` by default.
                To use `samplerate` or `resampy`, they should be installed manually::

                    $ pip install samplerate
                    $ pip install resampy

            .. note::
                When using ``res_type='polyphase'``, only integer sampling rates are
                supported.
    
    .. automethod:: __call__
    '''
    def __init__(self,origin_sampling_rate:int, target_sampling_rate:int, 
                 apply_to_baseline=False,
                 axis= -1,
                 scale:bool=False,
                 res_type:str='soxr_hq'):
        
        super(SetSamplingRate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.original_rate = origin_sampling_rate
        self.new_rate = target_sampling_rate
        self.axis = axis
        self.scale = scale
        self.res_type = res_type

    
    def apply(self,
        eeg: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # First, validate the audio buffer
        import lazy_loader as lazy
        # Lazy-load optional dependencies
        samplerate = lazy.load("samplerate")
        resampy = lazy.load("resampy")
        soxr = lazy.load('soxr')

        eeg = eeg.astype(np.float32)

        if self.original_rate == self.new_rate:
            return eeg

        ratio = float(self.new_rate) / self.original_rate

        n_samples = int(np.ceil(eeg.shape[self.axis] * ratio))

        if self.res_type in ("scipy", "fft"):
            EEG_res = scipy.signal.resample(eeg, n_samples, axis=self.axis)
        elif self.res_type == "polyphase":
            # For polyphase resampling, we need up- and down-sampling ratios
            # We can get those from the greatest common divisor of the rates
            # as long as the rates are integrable
            self.original_rate = int(self.original_rate)
            self.new_rate = int(self.new_rate)
            gcd = np.gcd(self.original_rate, self.new_rate)
            EEG_res = scipy.signal.resample_poly(
                eeg, self.new_rate // gcd, self.original_rate // gcd, axis=self.axis
            )
        elif self.res_type in (
            "linear",
            "zero_order_hold",
            "sinc_best",
            "sinc_fastest",
            "sinc_medium",
        ):
            # Use numpy to vectorize the resampler along the target axis
            # This is because samplerate does not support ndim>2 generally.
            EEG_res = np.apply_along_axis(
                samplerate.resample, axis=self.axis, arr=eeg, ratio=ratio, converter_type=self.res_type
            )
        elif self.res_type.startswith("soxr"):
            # Use numpy to vectorize the resampler along the target axis
            # This is because soxr does not support ndim>2 generally.
            EEG_res = np.apply_along_axis(
                soxr.resample,
                axis=self.axis,
                arr=eeg,
                in_rate=self.original_rate,
                out_rate=self.new_rate,
                quality=self.res_type,
            )
        else:
            EEG_res = resampy.resample(eeg, self.original_rate, self.new_rate, filter=self.res_type, axis=self.axis)

        if self.scale:
            EEG_res /= np.sqrt(ratio)

        # Match dtypes
        return np.asarray(EEG_res, dtype=eeg.dtype)

    
    @property
    def __repr__(self)->any :
        return  f'''{
                'original_sampling_rate': self.original_rate,
                'target_sampling_rate': self.new_rate,
                'apply_to_baseline':self.apply_to_baseline
                'axis': self.axis,
                'scale': self.scale,
                'res_type': self.res_type
            }'''
