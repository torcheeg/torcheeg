from typing import Dict, List, Union

import numpy as np

from ..base_transform import EEGTransform


class RearrangeElectrode(EEGTransform):
    r"""
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.RearrangeElectrode(
            source=['FP1', 'F3', 'F7'],
            target=['F3', 'F7', 'FP1', 'AF2'],
            missing='mean'
        )
        t(np.random.randn(3, 128)).shape
        >>> (4, 128)

    Args:
        source (list): The list of electrode names to be rearranged.
        target (list): The list of electrode names to be rearranged to.
        missing (str): The method to deal with missing electrodes. (default: :obj:`'random'`)

    .. automethod:: __call__
    """

    def __init__(self, source: List[str], target: List[str], missing: str = 'mean'):
        super(RearrangeElectrode, self).__init__()
        self.source = source
        self.target = target
        if not missing in ['random', 'zero', 'mean']:
            raise ValueError(
                f'Invalid missing method {missing}, should be one of [random, zero, mean].')
        self.missing = missing

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
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
