from typing import Dict, Tuple, Union

from scipy.signal import butter, hann, lfilter

import numpy as np

from ..base_transform import EEGTransform


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


class BandTransform(EEGTransform):
    def __init__(self,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandTransform, self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = butter_bandpass(low,
                                       high,
                                       fs=self.sampling_rate,
                                       order=self.order)
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'order': self.order,
                'band_dict': {...}
            })


class BandSignal(BandTransform):
    r'''
    A transform method to split the EEG signal into signals in different sub-bands.

    .. code-block:: python

        transform = BandSignal()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (4, 32, 128)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The differential entropy of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = butter(self.order, [low, high],
                              fs=self.sampling_rate,
                              btype="band")
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=0)


class BandDifferentialEntropy(BandTransform):
    r'''
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input. It is a widely accepted differential entropy calculation method by the community, which is often applied to the DEAP and DREAMER datasets. It is relatively easy to understand and has a smaller scale and more gradual changes than the :obj:`BandDifferentialEntropyV1` calculated based on average power spectral density. 

    - Related Paper: Fdez J, Guttenberg N, Witkowski O, et al. Cross-subject EEG-based emotion recognition through neural networks with stratified normalization[J]. Frontiers in neuroscience, 2021, 15: 626277.
    - Related Project: https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn/

    - Related Paper: Li D, Xie L, Chai B, et al. Spatial-frequency convolutional self-attention network for EEG emotion recognition[J]. Applied Soft Computing, 2022, 122: 108740.
    - Related Project: https://github.com/qeebeast7/SFCSAN/

    In most cases, choosing :obj:`BandDifferentialEntropy` and :obj:`BandDifferentialEntropyV1` does not make much difference. If you have other comments, please feel free to pull request.
    
    .. code-block:: python

        transform = BandDifferentialEntropy()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The differential entropy of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.std(eeg))


class BandDifferentialEntropyV1(EEGTransform):
    r'''
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input. This version calculates the differential entropy based on the relationship between the differential entropy and the average power spectral density, which is identical to the processing of the SEED dataset.

    - Related Paper: Shi L C, Jiao Y Y, Lu B L. Differential entropy feature for EEG-based vigilance estimation[C]//2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2013: 6627-6630.
    - Related Project: https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD

    .. code-block:: python

        transform = BandDifferentialEntropyV1()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 sampling_rate: int = 128,
                 fft_n: int = None,
                 num_window: int = 1,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandDifferentialEntropyV1,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate

        if fft_n is None:
            fft_n = self.sampling_rate

        self.fft_n = fft_n
        self.num_window = num_window
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        _, chunk_size = eeg.shape
        point_per_window = int(chunk_size // self.num_window)

        band_list = []

        for window_index in range(self.num_window):
            start_index, end_index = point_per_window * window_index, point_per_window * (
                window_index + 1)
            window_data = eeg[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=self.fft_n)
            energy_graph = np.abs(fft_data[:, 0:int(self.fft_n / 2)])

            for _, band in enumerate(self.band_dict.values()):
                start_index = int(
                    np.floor(band[0] / self.sampling_rate * self.fft_n))
                end_index = int(
                    np.floor(band[1] / self.sampling_rate * self.fft_n))
                band_ave_psd = np.mean(energy_graph[:, start_index -
                                                    1:end_index]**2,
                                       axis=1)
                # please refer to # https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD/blob/master/DE_PSD.py, which consider the relationship between DE and PSD to calculate DE.
                band_de = np.log2(100 * band_ave_psd)
                band_list.append(band_de)

        return np.stack(band_list, axis=-1)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The power spectral density of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'fft_n': self.fft_n,
                'num_window': self.num_window,
                'band_dict': {...}
            })


class BandPowerSpectralDensity(EEGTransform):
    r'''
    A transform method for calculating the power spectral density of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandPowerSpectralDensity()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The sampling rate of EEG signals in Hz. (default: :obj:`128`)
        fft_n (int): Computes the one-dimensional n-point discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT) algorithm. If set to None, it will automatically match sampling_rate. (default: :obj:`None`)
        num_window (int): Welch's method computes an estimate of the power spectral density by dividing the data into non-overlapping segments, where the num_window denotes the number of windows. (default: :obj:`1`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the power spectral density of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 sampling_rate: int = 128,
                 fft_n: int = None,
                 num_window: int = 1,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandPowerSpectralDensity,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate

        if fft_n is None:
            fft_n = self.sampling_rate

        self.fft_n = fft_n
        self.num_window = num_window
        self.band_dict = band_dict

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        _, chunk_size = eeg.shape
        point_per_window = int(chunk_size // self.num_window)

        band_list = []

        for window_index in range(self.num_window):
            start_index, end_index = point_per_window * window_index, point_per_window * (
                window_index + 1)
            window_data = eeg[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=self.fft_n)
            energy_graph = np.abs(fft_data[:, 0:int(self.fft_n / 2)])

            for _, band in enumerate(self.band_dict.values()):
                start_index = int(
                    np.floor(band[0] / self.sampling_rate * self.fft_n))
                end_index = int(
                    np.floor(band[1] / self.sampling_rate * self.fft_n))
                band_ave_psd = np.mean(energy_graph[:, start_index -
                                                    1:end_index]**2,
                                       axis=1)

                band_list.append(band_ave_psd)

        return np.stack(band_list, axis=-1)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The power spectral density of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'fft_n': self.fft_n,
                'num_window': self.num_window,
                'band_dict': {...}
            })


class BandMeanAbsoluteDeviation(BandTransform):
    r'''
    A transform method for calculating the mean absolute deviation of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandMeanAbsoluteDeviation()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the mean absolute deviation of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The mean absolute deviation of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return np.mean(np.abs(eeg - np.mean(eeg)))


class BandKurtosis(BandTransform):
    r'''
    A transform method for calculating the kurtosis of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandKurtosis()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the kurtosis of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The kurtosis of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n = len(eeg)
        ave1 = 0.0
        ave2 = 0.0
        ave4 = 0.0
        for eeg in eeg:
            ave1 += eeg
            ave2 += eeg**2
            ave4 += eeg**4
        ave1 /= n
        ave2 /= n
        ave4 /= n
        sigma = np.sqrt(ave2 - ave1**2)
        return ave4 / (sigma**4)


class BandSkewness(BandTransform):
    r'''
    A transform method for calculating the skewness of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandSkewness()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate of EEG signals in Hz. (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the skewness of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)

    .. automethod:: __call__
    '''
    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The skewness of several sub-bands for all electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n = len(eeg)
        ave1 = 0.0
        ave2 = 0.0
        ave3 = 0.0
        for eeg in eeg:
            ave1 += eeg
            ave2 += eeg**2
            ave3 += eeg**3
        ave1 /= n
        ave2 /= n
        ave3 /= n
        sigma = np.sqrt(ave2 - ave1**2)
        return (ave3 - 3 * ave1 * sigma**2 - ave1**3) / (sigma**3)
