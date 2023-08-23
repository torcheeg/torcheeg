from typing import Dict, List, Union

import numpy as np

from ..base_transform import EEGTransform


class RearrangeElectrode(EEGTransform):
    r'''
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        transform = RearrangeElectrode(
            source=['FP1', 'F3', 'F7'],
            target=['F3', 'F7', 'FP1', 'AF2'],
            missing='mean'
        )
        transform(eeg=np.random.randn(3, 128))['eeg'].shape
        >>> (4, 128)

    Args:
        source (list): The list of electrode names to be rearranged.
        target (list): The list of electrode names to be rearranged to.
        missing (str): The method to deal with missing electrodes. (default: :obj:`'random'`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 source: List[str],
                 target: List[str],
                 missing: str = 'mean',
                 apply_to_baseline: bool = False):
        super(RearrangeElectrode,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.source = source
        self.target = target

        assert missing in [
            'random', 'zero', 'mean'
        ], f'Invalid missing method {missing}, should be one of [random, zero, mean].'

        self.missing = missing

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
            np.ndarray: The output signals with the shape of [number of picked electrodes, number of data points].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].

        Returns:
            np.ndarray: The output signals with the shape of [number of rearranged electrodes, number of data points].
        '''
        output = np.zeros((len(self.target), eeg.shape[1]))
        for i, target in enumerate(self.target):
            if target in self.source:
                output[i] = eeg[self.source.index(target)]
            else:
                if self.missing == 'random':
                    output[i] = np.random.randn(eeg.shape[1])
                elif self.missing == 'zero':
                    output[i] = np.zeros(eeg.shape[1])
                elif self.missing == 'mean':
                    output[i] = np.mean(eeg, axis=0)
        return output

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'source': self.source,
                'target': self.target,
                'missing': self.missing
            })
