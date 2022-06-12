from typing import Dict, Union

import torch

from ..base_transform import EEGTransform


class RandomNoise(EEGTransform):
    '''
    Add random noise conforming to the normal distribution on the EEG signal.
    
    .. code-block:: python

        transform = RandomNoise(p=0.5)
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        mean (float): The mean of the normal distribution of noise. (defualt: :obj:`0.0`)
        std (float): The standard deviation of the normal distribution of noise. (defualt: :obj:`0.0`)
        p (float): Probability of adding noise to EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no noise is added to every sample and 1.0 means that noise is added to every sample. (defualt: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 mean: float = 0.0,
                 std: float = 1.0,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomNoise, self).__init__(apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self,
                 *args,
                 eeg: torch.Tensor,
                 baseline: Union[torch.Tensor, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch.Tensor: The output EEG signal after adding random noise.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.p < torch.rand(1):
            return eeg
        noise = torch.randn_like(eeg)
        noise = (noise + self.mean) * self.std
        return eeg + noise


class RandomMask(EEGTransform):
    '''
    Overlay the EEG signal using a random mask, and the value of the overlaid data points was set to 0.0.
    
    .. code-block:: python

        transform = RandomMask()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        ratio (float): The proportion of data points covered by the mask out of all data points for each EEG signal sample. (defualt: :obj:`0.5`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (defualt: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 ratio: float = 0.5,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomMask, self).__init__(apply_to_baseline=apply_to_baseline)
        self.ratio = ratio
        self.p = p

    def __call__(self,
                 *args,
                 eeg: torch.Tensor,
                 baseline: Union[torch.Tensor, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch.Tensor: The output EEG signal after applying a random mask.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.p < torch.rand(1):
            return eeg
        mask = torch.rand_like(eeg)
        mask = (mask < self.ratio).to(eeg.dtype)
        return eeg * mask
