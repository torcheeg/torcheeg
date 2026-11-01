from typing import Dict, Tuple, Union

import numpy as np
from scipy.signal import hilbert

from ..base_transform import EEGTransform


class CorrelationTransform(EEGTransform):
    def __init__(self):
        super(CorrelationTransform, self).__init__()

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
        if not (len(eeg.shape) == 2):
            raise ValueError(
                'The input of CorrelationTransform must have 2 or 3 dimensions, which represent band number [optional], electrode number and time point number.'
            )

        num_item = eeg.shape[0]
        matrix = np.zeros((num_item, num_item))
        for i in range(num_item):
            for j in range(num_item):
                matrix[i][j] = self.opt(eeg[i], eeg[j])
        return matrix[np.newaxis, ...]

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Method opt is not implemented in class "
            + self.__class__.__name__
        )


class PearsonCorrelation(CorrelationTransform):
    r"""
    A transform method to calculate the correlation coefficients between the EEG signals of different electrodes.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.PearsonCorrelation()
        t(np.random.randn(32, 128)).shape
        >>> (1, 32, 32)

    Args:
        absolute (bool): Whether to take the absolute value of the correlation coefficient. (default: :obj:`128`)


    .. automethod:: __call__
    """

    def __init__(self, absolute: bool = False):
        super(PearsonCorrelation, self).__init__()
        self.absolute = absolute

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.corrcoef(x, y)[0][1]
        if self.absolute:
            out = np.absolute(out)
        return out

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'absolute': self.absolute})


class PhaseLockingValue(CorrelationTransform):
    r"""
    Calculate standard phase-locking values (PLV) between electrodes over the
    time axis of each EEG window.

    For Hilbert analytic phases :math:`\phi_x(t)` and :math:`\phi_y(t)`,
    the pairwise feature is

    .. math::

        \mathrm{PLV}(x, y) = \left|\frac{1}{T}\sum_{t=1}^{T}
        e^{i(\phi_x(t)-\phi_y(t))}\right|.

    Values lie in [0, 1]. A constant phase difference gives 1, including a
    nonzero constant difference. Each sample contributes a unit-length phase
    vector, so instantaneous amplitudes do not weight the average.

    .. code-block:: python

        from torcheeg import transforms
        import numpy as np

        t = transforms.PhaseLockingValue()
        t(np.random.randn(32, 128)).shape
        >>> (1, 32, 32)

    Args:


    .. automethod:: __call__
    """

    def __init__(self):
        super(PhaseLockingValue, self).__init__()

    def opt(self, x: np.ndarray, y: np.ndarray) -> float:
        x_hill = hilbert(x)
        y_hill = hilbert(y)
        for analytic in (x_hill, y_hill):
            if not np.isfinite(analytic).all() or np.any(np.abs(analytic) == 0):
                raise ValueError(
                    'PhaseLockingValue requires finite analytic signals with '
                    'nonzero amplitude at every sample.')
        phase_difference = (np.angle(x_hill).astype(np.float64) -
                            np.angle(y_hill).astype(np.float64))
        value = np.abs(np.mean(np.exp(1j * phase_difference)))
        return float(np.clip(value, 0.0, 1.0))
