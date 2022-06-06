from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

from scipy.signal import butter, lfilter, welch

import numpy as np


class BandTransform(metaclass=ABCMeta):
    def __init__(self,
                 frequency: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 }):
        self.frequency = frequency
        self.order = order
        self.band_dict = band_dict

    def __call__(self, x: np.ndarray) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in x:
                b, a = butter(self.order, [low, high], fs=self.frequency, btype="band")
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    @abstractmethod
    def opt(self, x: np.ndarray) -> np.ndarray:
        ...


class BandDifferentialEntropy(BandTransform):
    r'''
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandDifferentialEntropy()
        transform(torch.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        frequency (int): The sample frequency in Hz. (defualt: :obj:`128`)
        order (int): The order of the filter. (defualt: :obj:`5`)
        band_dict: (dict): Band name and the critical frequency or frequencies. By default, the differential entropy of the four subbands, theta, alpha, beta and gamma, is calculated. (defualt: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of subbands]: The differential entropy of several subbands for all electrodes.
        '''
        return super().__call__(x)

    def opt(self, x: np.ndarray) -> np.ndarray:
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.std(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BandPowerSpectralDensity:
    r'''
    A transform method for calculating the power spectral density of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandPowerSpectralDensity()
        transform(torch.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        frequency (int): The sample frequency in Hz. (defualt: :obj:`128`)
        window (int): Welch's method computes an estimate of the power spectral density by dividing the data into overlapping segments, where the window denotes length of each segment. (defualt: :obj:`128`)
        order (int): The order of the filter. (defualt: :obj:`5`)
        band_dict: (dict): Band name and the critical frequency or frequencies. By default, the power spectral density of the four subbands, theta, alpha, beta and gamma, is calculated. (defualt: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 frequency: int = 128,
                 window_size: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 }):
        self.frequency = frequency
        self.window_size = window_size
        self.order = order
        self.band_dict = band_dict

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of subbands]: The power spectral density of several subbands for all electrodes.
        '''
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in x:
                freqs, psd = welch(c, self.frequency, nperseg=self.window_size, scaling='density')

                index_min = np.argmax(np.round(freqs) > low) - 1
                index_max = np.argmax(np.round(freqs) > high)

                c_list.append(psd[index_min:index_max].mean())
            band_list.append(np.array(c_list))
        return np.stack(band_list, axis=-1)


class BandMeanAbsoluteDeviation(BandTransform):
    r'''
    A transform method for calculating the mean absolute deviation of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandMeanAbsoluteDeviation()
        transform(torch.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        frequency (int): The sample frequency in Hz. (defualt: :obj:`128`)
        order (int): The order of the filter. (defualt: :obj:`5`)
        band_dict: (dict): Band name and the critical frequency or frequencies. By default, the mean absolute deviation of the four subbands, theta, alpha, beta and gamma, is calculated. (defualt: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of subbands]: The mean absolute deviation of several subbands for all electrodes.
        '''
        return super().__call__(x)

    def opt(self, x: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(x - np.mean(x)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BandKurtosis(BandTransform):
    r'''
    A transform method for calculating the kurtosis of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandKurtosis()
        transform(torch.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        frequency (int): The sample frequency in Hz. (defualt: :obj:`128`)
        order (int): The order of the filter. (defualt: :obj:`5`)
        band_dict: (dict): Band name and the critical frequency or frequencies. By default, the kurtosis of the four subbands, theta, alpha, beta and gamma, is calculated. (defualt: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of subbands]: The kurtosis of several subbands for all electrodes.
        '''
        return super().__call__(x)

    def opt(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        ave1 = 0.0
        ave2 = 0.0
        ave4 = 0.0
        for x in x:
            ave1 += x
            ave2 += x**2
            ave4 += x**4
        ave1 /= n
        ave2 /= n
        ave4 /= n
        sigma = np.sqrt(ave2 - ave1**2)
        return ave4 / (sigma**4)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BandSkewness(BandTransform):
    r'''
    A transform method for calculating the skewness of EEG signals in several sub-bands with EEG signals as input.

    .. code-block:: python

        transform = BandSkewness()
        transform(torch.randn(32, 128)).shape
        >>> (32, 4)

    Args:
        frequency (int): The sample frequency in Hz. (defualt: :obj:`128`)
        order (int): The order of the filter. (defualt: :obj:`5`)
        band_dict: (dict): Band name and the critical frequency or frequencies. By default, the skewness of the four subbands, theta, alpha, beta and gamma, is calculated. (defualt: :obj:`{...}`)
    
    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of subbands]: The skewness of several subbands for all electrodes.
        '''
        return super().__call__(x)

    def opt(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        ave1 = 0.0
        ave2 = 0.0
        ave3 = 0.0
        for x in x:
            ave1 += x
            ave2 += x**2
            ave3 += x**3
        ave1 /= n
        ave2 /= n
        ave3 /= n
        sigma = np.sqrt(ave2 - ave1**2)
        return (ave3 - 3 * ave1 * sigma**2 - ave1**3) / (sigma**3)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
