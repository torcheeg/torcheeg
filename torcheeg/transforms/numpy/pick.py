from typing import Dict, List, Union

import numpy as np

from ..base_transform import EEGTransform


class PickElectrode(EEGTransform):
    r'''
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        transform = PickElectrode(PickElectrode.to_index_list(
            ['FP1', 'AF3', 'F3', 'F7',
             'FC5', 'FC1', 'C3', 'T7',
             'CP5', 'CP1', 'P3', 'P7',
             'PO3','O1', 'FP2', 'AF4',
             'F4', 'F8', 'FC6', 'FC2',
             'C4', 'T8', 'CP6', 'CP2',
             'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST))
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (28, 128)

    Args:
        pick_list (np.ndarray): Selected electrode list. Should consist of integers representing the corresponding electrode indices. :obj:`to_index_list` can be used to obtain an index list when we only know the names of the electrode and not their indices.
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self, pick_list: List[int], apply_to_baseline: bool = False):
        super(PickElectrode, self).__init__(apply_to_baseline=apply_to_baseline)
        self.pick_list = pick_list

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
        assert max(self.pick_list) < eeg.shape[
            0], f'The index {max(self.pick_list)} of the specified electrode is out of bounds {eeg.shape[0]}.'
        return eeg[self.pick_list]

    @staticmethod
    def to_index_list(electrode_list: List[str],
                      dataset_electrode_list: List[str],
                      strict_mode=False) -> List[int]:
        r'''
        Args:
            electrode_list (list): picked electrode name, consisting of strings.
            dataset_electrode_list (list): The description of the electrode information contained in the EEG signal in the dataset, consisting of strings. For the electrode position information, please refer to constants grouped by dataset :obj:`datasets.constants`.
            strict_mode: (bool): Whether to use strict mode. In strict mode, unmatched picked electrode names are thrown as errors. Otherwise, unmatched picked electrode names are automatically ignored. (default: :obj:`False`)
        Returns:
            list: Selected electrode list, consisting of integers representing the corresponding electrode indices.
        '''
        dataset_electrode_dict = dict(
            zip(dataset_electrode_list,
                list(range(len(dataset_electrode_list)))))
        if strict_mode:
            return [
                dataset_electrode_dict[electrode]
                for electrode in electrode_list
            ]
        return [
            dataset_electrode_dict[electrode] for electrode in electrode_list
            if electrode in dataset_electrode_dict
        ]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'pick_list': [...]
        })