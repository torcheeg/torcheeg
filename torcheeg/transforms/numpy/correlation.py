from typing import Dict, Tuple, Union

from scipy.signal import hilbert

import numpy as np

from ..base_transform import EEGTransform


class CorrelationTransform(EEGTransform):
    def __init__(self, apply_to_baseline: bool = False):
        super(CorrelationTransform,
              self).__init__(apply_to_baseline=apply_to_baseline)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        if len(eeg.shape) == 3:
            band_list = []
            num_item = eeg.shape[1]

            for band_eeg in eeg:
                matrix = np.zeros((num_item, num_item))
                for i in range(num_item):
                    for j in range(num_item):
                        matrix[i][j] = self.opt(band_eeg[i], band_eeg[j])
                band_list.append(matrix)
            return np.stack(band_list, axis=0)
        assert len(
            eeg.shape
        ) == 2, 'The input of CorrelationTransform must have 2 or 3 dimensions, which represent band number [optional], electrode number and time point number.'

        num_item = eeg.shape[0]
        matrix = np.zeros((num_item, num_item))
        for i in range(num_item):
            for j in range(num_item):
                matrix[i][j] = self.opt(eeg[i], eeg[j])
        return matrix[np.newaxis, ...]

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PearsonCorrelation(CorrelationTransform):
    r'''
    A transform method to calculate the correlation coefficients between the EEG signals of different electrodes.

    .. code-block:: python

        transform = BandSignal()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (1, 32, 32)

    Args:
        absolute (bool): Whether to take the absolute value of the correlation coefficient. (default: :obj:`128`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self, absolute: bool = False, apply_to_baseline: bool = False):
        super(PearsonCorrelation,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.absolute = absolute

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
            np.ndarray[number of electrodes, number of electrodes]: The correlation coefficients between EEG signals of different electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.corrcoef(x, y)[0][1]
        if self.absolute:
            out = np.absolute(out)
        return out

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'absolute': self.absolute})


class PhaseLockingCorrelation(CorrelationTransform):
    r'''
    A transform method to calculate the phase locking values between the EEG signals of different electrodes.

    .. code-block:: python

        transform = BandSignal()
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (1, 32, 32)

    Args:
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self, apply_to_baseline: bool = False):
        super(PhaseLockingCorrelation,
              self).__init__(apply_to_baseline=apply_to_baseline)

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
            np.ndarray[number of electrodes, number of electrodes]: The phase locking values between EEG signals of different electrodes.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_hill = hilbert(x)
        y_hill = hilbert(y)
        pdt = (np.inner(x_hill, np.conj(y_hill)) / (np.sqrt(
            np.inner(x_hill, np.conj(x_hill)) *
            np.inner(y_hill, np.conj(y_hill)))))
        return np.angle(pdt)
