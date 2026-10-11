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

        from torcheeg import transforms

        t = transforms.PearsonCorrelation()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
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
    Calculate the phase angle of the normalized complex correlation between
    the Hilbert analytic signals of different electrodes, in radians.

    This legacy transform preserves its original numerical behavior. It does
    not calculate the standard phase-locking value (PLV): its output is a
    signed angle, rather than a strength in [0, 1]. Use
    :class:`PhaseLockingValue` for standard PLV features.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.PhaseLockingCorrelation()
        t(eeg=np.random.randn(32, 128))['eeg'].shape
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
            Dict[str, np.ndarray]: Correlation phase-angle matrices in radians
            under the ``eeg`` key. Each matrix has shape
            [1, electrodes, electrodes] for a 2D input or
            [bands, electrodes, electrodes] for a 3D input. The same rule
            applies to ``baseline`` when baseline transformation is enabled.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def opt(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_hill = hilbert(x)
        y_hill = hilbert(y)
        pdt = (np.inner(x_hill, np.conj(y_hill)) / (np.sqrt(
            np.inner(x_hill, np.conj(x_hill)) *
            np.inner(y_hill, np.conj(y_hill)))))
        return np.angle(pdt)


class PhaseLockingValue(CorrelationTransform):
    r'''
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

    Input shape is [electrodes, time points] or [bands, electrodes, time
    points]. Output shape is [1, electrodes, electrodes] or
    [bands, electrodes, electrodes], respectively. The input must contain
    finite real numbers and have nonempty axes. Zero analytic amplitude at
    any sample raises :class:`ValueError`, because its phase is undefined.
    A zero real sample alone is valid when its analytic amplitude is nonzero.

    Filter signals to the frequency band of interest before applying this
    transform. No filtering is performed here. See
    :doc:`/phase_locking_value` for the mathematical properties and the
    distinction from :class:`PhaseLockingCorrelation`.

    .. code-block:: python

        from torcheeg import transforms
        import numpy as np

        theta = 2 * np.pi * np.arange(128) / 128
        eeg = np.stack([np.cos(8 * theta), np.cos(8 * theta + np.pi / 2)])
        transforms.PhaseLockingValue()(eeg=eeg)['eeg']
        # All entries are approximately 1, with shape (1, 2, 2).

    Args:
        apply_to_baseline (bool): Transform the baseline with the same rule
            when it is provided. Otherwise, leave it unchanged.
            (default: :obj:`False`)

    .. automethod:: __call__
    '''

    def __init__(self, apply_to_baseline: bool = False):
        super(PhaseLockingValue,
              self).__init__(apply_to_baseline=apply_to_baseline)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): EEG with shape [electrodes, time points] or
                [bands, electrodes, time points].
            baseline (np.ndarray, optional): Baseline transformed only when
                ``apply_to_baseline=True``.

        Returns:
            Dict[str, np.ndarray]: Pairwise PLV matrices under the ``eeg`` key,
            and under ``baseline`` when baseline transformation is enabled.

        Raises:
            ValueError: If transformed input is empty, complex, nonfinite, or
                has a zero-amplitude analytic sample.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        eeg = np.asarray(eeg)
        if eeg.ndim not in (2, 3) or any(size == 0 for size in eeg.shape):
            raise ValueError(
                'PhaseLockingValue requires nonempty [electrodes, time points] '
                'or [bands, electrodes, time points] input.')
        self._validate_signal(eeg)
        return super().apply(eeg, **kwargs)

    @staticmethod
    def _validate_signal(signal: np.ndarray):
        if (not np.issubdtype(signal.dtype, np.number)
                or np.iscomplexobj(signal) or not np.isfinite(signal).all()):
            raise ValueError('PhaseLockingValue requires finite real signals.')

    def opt(self, x: np.ndarray, y: np.ndarray) -> float:
        phases = []
        for signal in (np.asarray(x), np.asarray(y)):
            if signal.ndim != 1 or signal.size == 0:
                raise ValueError('PhaseLockingValue requires nonempty 1D signals.')
            self._validate_signal(signal)
            analytic = hilbert(signal)
            if (not np.isfinite(analytic).all()
                    or np.any(np.abs(analytic) == 0)):
                raise ValueError(
                    'PhaseLockingValue requires finite analytic signals with '
                    'nonzero amplitude at every sample.')
            phases.append(np.angle(analytic).astype(np.float64))
        if phases[0].shape != phases[1].shape:
            raise ValueError('PhaseLockingValue requires signals of equal length.')
        phase_vectors = np.exp(1j * (phases[0] - phases[1]))
        value = np.abs(np.mean(phase_vectors))
        return float(np.clip(value, 0.0, 1.0))
