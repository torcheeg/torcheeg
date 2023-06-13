from typing import Dict, Tuple, Union
from scipy.signal import butter, lfilter, hann

import numpy as np

from ..base_transform import EEGTransform

np.seterr(all="ignore")


def butter_bandpass(low_cut, high_cut, fs, order=3):
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


def embed_seq(time_series, tau, embedding_dimension):
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (typed_time_series.size - tau * (embedding_dimension - 1),
             embedding_dimension)

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(typed_time_series,
                                           shape=shape,
                                           strides=strides)


class BandApproximateEntropy(BandTransform):
    r'''
    A transform method for calculating the approximate entropy of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/entropy.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandApproximateEntropy()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        M (int): A positive integer represents the length of each compared run of data. (default: :obj:`5`)
        R (float): A positive real number specifies a filtering level. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 M: int = 5,
                 R: float = 1.0,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandApproximateEntropy,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.M = M
        self.R = R

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
        N = len(eeg)

        Em = embed_seq(eeg, 1, self.M)
        A = np.tile(Em, (len(Em), 1, 1))
        B = np.transpose(A, [1, 0, 2])
        D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
        InRange = np.max(D, axis=2) <= self.R

        # Probability that random M-sequences are in range
        Cm = InRange.mean(axis=0)

        # M+1-sequences in range if M-sequences are in range & last values are close
        Dp = np.abs(
            np.tile(eeg[self.M:], (N - self.M, 1)) -
            np.tile(eeg[self.M:], (N - self.M, 1)).T)

        Cmp = np.logical_and(Dp <= self.R, InRange[:-1, :-1]).mean(axis=0)

        Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))

        Ap_En = (Phi_m - Phi_mp) / (N - self.M)

        return Ap_En

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'M': self.M, 'R': self.R})


class BandSampleEntropy(BandTransform):
    r'''
    A transform method for calculating the sample entropy of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/entropy.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandSampleEntropy()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        M (int): A positive integer represents the length of each compared run of data. (default: :obj:`5`)
        R (float): A positive real number specifies a filtering level. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 M: int = 5,
                 R: float = 1.0,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandSampleEntropy,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.M = M
        self.R = R

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
        N = len(eeg)

        Em = embed_seq(eeg, 1, self.M)
        A = np.tile(Em, (len(Em), 1, 1))
        B = np.transpose(A, [1, 0, 2])
        D = np.abs(A - B)
        InRange = np.max(D, axis=2) <= self.R
        np.fill_diagonal(InRange, 0)

        Cm = InRange.sum(axis=0)
        Dp = np.abs(
            np.tile(eeg[self.M:], (N - self.M, 1)) -
            np.tile(eeg[self.M:], (N - self.M, 1)).T)

        Cmp = np.logical_and(Dp <= self.R, InRange[:-1, :-1]).sum(axis=0)

        Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

        return Samp_En

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'M': self.M, 'R': self.R})


class BandSVDEntropy(BandTransform):
    r'''
    A transform method for calculating the SVD entropy of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/entropy.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandSVDEntropy()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        Tau (int): A positive integer represents the embedding time delay which controls the number of time periods between elements of each of the new column vectors. (default: :obj:`1`)
        DE (int): A positive integer represents the ength of the embedding dimension. (default: :obj:`1`)
        W (np.ndarray, optional): A list of normalized singular values of the embedding matrix (can be preset for speeding up). (default: :obj:`None`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 Tau: int = 1,
                 DE: int = 1,
                 W: Union[np.ndarray, None] = None,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandSVDEntropy,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.Tau = Tau
        self.DE = DE
        self.W = W

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
        if self.W is None:
            Y = embed_seq(eeg, self.Tau, self.DE)
            W = np.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values
        else:
            W = self.W
        return -1 * sum(W * np.log(W))

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'Tau': self.Tau,
            'DE': self.DE,
            'W': [...]
        })


class BandDetrendedFluctuationAnalysis(BandTransform):
    r'''
    A transform method for calculating the detrended fluctuation analysis (DFA) of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/detrended_fluctuation_analysis.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandDetrendedFluctuationAnalysis()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        Ave (float, optional): The average value of the time series. (default: :obj:`None`)
        L (List[np.array]): Box sizes to partition/slice/segment the integrated sequence into boxes. At least two boxes are needed, and it should be a list of integers in ascending order. (default: :obj:`np.array`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 Ave: Union[float, None] = None,
                 L: Union[np.ndarray, None] = None,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandDetrendedFluctuationAnalysis,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.Ave = Ave
        self.L = L

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
        eeg = np.array(eeg)

        if self.Ave is None:
            Ave = np.mean(eeg)
        else:
            Ave = self.Ave

        Y = np.cumsum(eeg)
        Y -= Ave

        if self.L is None:
            L = np.floor(
                len(eeg) * 1 /
                (2**np.array(list(range(4,
                                        int(np.log2(len(eeg))) - 4)))))
        else:
            L = self.L

        F = np.zeros(len(L))

        for i in range(0, len(L)):
            n = int(L[i])
            assert n, 'Time series is too short while the box length is too big.'
            for j in range(0, len(eeg), n):
                if j + n < len(eeg):
                    c = list(range(j, j + n))
                    c = np.vstack([c, np.ones(n)]).T
                    y = Y[j:j + n]
                    F[i] += np.linalg.lstsq(c, y)[1]
            F[i] /= ((len(eeg) / n) * n)
        F = np.sqrt(F)

        Alpha = np.linalg.lstsq(np.vstack([np.log(L),
                                           np.ones(len(L))]).T,
                                np.log(F),
                                rcond=None)[0][0]

        return Alpha

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'Ave': self.Ave, 'L': [...]})


class BandHiguchiFractalDimension(BandTransform):
    r'''
    A transform method for calculating the higuchi fractal dimension (HFD) of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/fractal_dimension.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandHiguchiFractalDimension()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        K_max (int): The max number of new self-similar time series applying Higuchi's algorithm. (default: :obj:`6`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 K_max: int = 6,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandHiguchiFractalDimension,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.K_max = K_max

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
        L = []
        x = []
        N = len(eeg)
        for k in range(1, self.K_max):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(eeg[m + i * k] - eeg[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])

        (p, _, _, _) = np.linalg.lstsq(x, L, rcond=None)
        return p[0]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'K_max': self.K_max})


class BandHjorth(BandTransform):
    r'''
    A transform method for calculating the hjorth mobility/complexity of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/hjorth_mobility_complexity.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandHjorth()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        D (np.ndarray, optional): The first order differential sequence of the time series (can be preset for speeding up). (default: :obj:`None`)
        mode (str): Options include mobility, complexity, and both, which are used to calculate hjorth mobility, hjorth complexity, and concatenate the two, respectively. (default: :obj:`mobility`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 D: Union[np.ndarray, None] = None,
                 mode: str = 'mobility',
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandHjorth, self).__init__(sampling_rate=sampling_rate,
                                         order=order,
                                         band_dict=band_dict,
                                         apply_to_baseline=apply_to_baseline)
        self.D = D
        self.mode = mode

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
        if self.D is None:
            D = np.diff(eeg)
            D = D.tolist()
        else:
            D = self.D

        D.insert(0, eeg[0])  # pad the first difference
        D = np.array(D)

        n = len(eeg)

        M2 = float(sum(D**2)) / n
        TP = sum(np.array(eeg)**2)
        M4 = 0
        for i in range(1, len(D)):
            M4 += (D[i] - D[i - 1])**2
        M4 = M4 / n

        mobility = np.sqrt(M2 / TP)
        complexity = np.sqrt(float(M4) * TP / M2 / M2)
        if self.mode == 'mobility':
            return mobility
        elif self.mode == 'complexity':
            return complexity
        elif self.mode == 'both':
            return np.concatenate(mobility, complexity)
        else:
            raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'D': [...]})


class BandHurst(BandTransform):
    r'''
    A transform method for calculating the hurst exponent of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/hurst.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandHurst()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    If the output H=0.5,the behavior of the EEG signals is similar to random walk. If H<0.5, the EEG signals cover less "distance" than a random walk, vice verse.
    
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
        eeg = np.array(eeg)
        N = eeg.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(eeg)
        Ave_T = Y / T

        S_T = np.zeros(N)
        R_T = np.zeros(N)

        for i in range(N):
            S_T[i] = np.std(eeg[:i + 1])
            eeg_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(eeg_T[:i + 1])

        R_S = R_T / S_T
        R_S = np.log(R_S)[1:]
        n = np.log(T)[1:]
        A = np.column_stack((n, np.ones(n.size)))
        [m, c] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = m
        return H


class BandPetrosianFractalDimension(BandTransform):
    r'''
    A transform method for calculating the petrosian fractal dimension (PFD) of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/fractal_dimension.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandHiguchiFractalDimension()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        D (np.ndarray, optional): The first order differential sequence of the time series (can be preset for speeding up). (default: :obj:`None`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 D: Union[np.ndarray, None] = None,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandPetrosianFractalDimension,
              self).__init__(sampling_rate=sampling_rate,
                             order=order,
                             band_dict=band_dict,
                             apply_to_baseline=apply_to_baseline)
        self.D = D

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
        if self.D is None:
            D = np.diff(eeg)
            D = D.tolist()
        else:
            D = self.D
        N_delta = 0
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(eeg)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'D': [...]})


def bin_power(X, band, sampling_rate):
    C = np.fft.fft(X)
    C = abs(C)
    power = np.zeros(len(band) - 1)
    for Freq_Index in range(0, len(band) - 1):
        Freq = float(band[Freq_Index])
        Next_Freq = float(band[Freq_Index + 1])
        power[Freq_Index] = sum(
            C[int(np.floor(Freq / sampling_rate *
                           len(X))):int(np.floor(Next_Freq / sampling_rate *
                                                 len(X)))])
    power_ratio = power / sum(power)
    return power, power_ratio


class BandBinPower(EEGTransform):
    r'''
    A transform method for calculating the power of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/fractal_dimension.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandBinPower()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        sampling_rate (int): The original sampling rate in Hz (default: :obj:`128`)
        order (int): The order of the filter. (default: :obj:`5`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
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
        super(BandBinPower, self).__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

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

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_a_b_list = []
        for band_a_b in self.band_dict.values():
            band_a_b_list += band_a_b
        band = sorted(list(set(band_a_b_list)))
        c_list = []
        for c in eeg:
            c_list.append(bin_power(c, band, self.sampling_rate)[1])
        return np.array(c_list)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'order': self.order,
                'band_dict': {...},
            })


class BandSpectralEntropy(EEGTransform):
    r'''
    A transform method for calculating the spectral entropy of EEG signals in several sub-bands with EEG signals as input. We revised part of the implementation in PyEEG to fit the TorchEEG pipeline.

    - Paper: Bao F S, Liu X, Zhang C. PyEEG: an open source python module for EEG/MEG feature extraction[J]. Computational intelligence and neuroscience, 2011, 2011.
    - URL: https://www.hindawi.com/journals/cin/2011/406391/
    - Related Project: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/entropy.py

    Please cite the above paper if you use this module.

    .. code-block:: python

        transform = BandSampleEntropy()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        power_ratio (np.ndarray, optional): A list of normalized signal power in the set of sub-bands (can be preset for speeding up). (default: :obj:`None`)
        band_dict: (dict): Band name and the critical sampling rate or frequencies. By default, the differential entropy of the four sub-bands, theta, alpha, beta and gamma, is calculated. (default: :obj:`{...}`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 power_ratio: Union[np.ndarray, None] = None,
                 sampling_rate: int = 128,
                 order: int = 5,
                 band_dict: Dict[str, Tuple[int, int]] = {
                     "theta": [4, 8],
                     "alpha": [8, 14],
                     "beta": [14, 31],
                     "gamma": [31, 49]
                 },
                 apply_to_baseline: bool = False):
        super(BandSpectralEntropy,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.power_ratio = power_ratio
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

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

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_a_b_list = []
        for band_a_b in self.band_dict.values():
            band_a_b_list += band_a_b
        band = sorted(list(set(band_a_b_list)))
        c_list = []
        for c in eeg:
            if self.power_ratio is None:
                power, power_ratio = bin_power(c, band, self.sampling_rate)
            else:
                power_ratio = self.power_ratio
            spectral_entropy = 0
            for i in range(0, len(power_ratio) - 1):
                spectral_entropy += power_ratio[i] * np.log(power_ratio[i])
            spectral_entropy /= np.log(len(power_ratio))
            c_list.append([-1 * spectral_entropy])
        return np.array(c_list)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'power_ratio': [...],
                'sampling_rate': self.sampling_rate,
                'order': self.order,
                'band_dict': {...},
            })