from typing import Dict, Union, List

from ..base_transform import EEGTransform


class BaselineRemoval(EEGTransform):
    r'''
    A transform method to subtract the baseline signal (the signal recorded before the emotional stimulus), the nosie signal is removed from the emotional signal unrelated to the emotional stimulus.
    
    TorchEEG recommends using this class in online_transform for higher processing speed. Even though, this class is also supported in offline_transform. Usually, the baseline needs the same transformation as the experimental signal, please add :obj:`apply_to_baseline=True` to all transforms before this operation to ensure that the transformation is performed on the baseline signal

    .. code-block:: python

        transform = Compose([
            BandDifferentialEntropy(apply_to_baseline=True),
            ToTensor(apply_to_baseline=True),
            BaselineRemoval(),
            ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])

        transform(eeg=np.random.randn(32, 128), baseline=np.random.randn(32, 128))['eeg'].shape
        >>> (4, 9, 9)
    
    .. automethod:: __call__
    '''
    def __init__(self):
        super(BaselineRemoval, self).__init__(apply_to_baseline=False)

    def __call__(self, *args, eeg: any, baseline: Union[any, None] = None, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            eeg (any): The input EEG signal.
            baseline (any) : The corresponding baseline signal.

        Returns:
            any: The transformed result after removing the baseline signal.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: any, **kwargs) -> any:
        if kwargs['baseline'] is None:
            return eeg

        assert kwargs[
            'baseline'].shape == eeg.shape, 'The baseline needs to change to the same shape as the input signal, please check if the `transform` is correct.'
        return eeg - kwargs['baseline']

    @property
    def targets_as_params(self) -> List[str]:
        return ['baseline']

    def get_params_dependent_on_targets(self, params: Dict[str, any]) -> Dict[str, any]:
        return {'baseline': params['baseline']}
