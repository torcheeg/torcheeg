from typing import Dict, List, Union

import numpy as np
import torch

from ..base_transform import EEGTransform


class Contrastive(EEGTransform):
    r'''
    To allow efficient training, we need to prepare the data loading such that we sample two different, random augmentations for each EEG in the batch. The easiest way to do this is by creating a transformation that, when being called, applies a set of data augmentations to an EEG twice.

    .. code-block:: python

        transform = Contrastive(RandomNoise(), num_views=2)
        transform(eeg=torch.randn(32, 128))['eeg'][0].shape
        >>> (32, 128)
        transform(eeg=torch.randn(32, 128))['eeg'][1].shape
        >>> (32, 128)

    .. automethod:: __call__
    '''

    def __init__(self,
                 transform: List[EEGTransform],
                 num_views: int = 2,
                 apply_to_baseline: bool = False):
        super(Contrastive, self).__init__(apply_to_baseline=apply_to_baseline)
        self.transform = transform
        self.num_views = num_views

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (torch.Tensor): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            list of torch.Tensor: The transformed results of multiple views.
        '''
        if args:
            raise KeyError("Please pass data as named parameters.")

        target_kwargs = {}
        non_target_kwargs = {}

        for _ in range(self.num_views):
            new_kwargs_t = self.transform(**kwargs)
            for new_kwargs_key, new_kwargs_value in new_kwargs_t.items():
                if not new_kwargs_key in self.targets:
                    non_target_kwargs[new_kwargs_key] = new_kwargs_value
                    continue
                if not new_kwargs_key in target_kwargs:
                    target_kwargs[new_kwargs_key] = []
                target_kwargs[new_kwargs_key].append(new_kwargs_value)

        target_kwargs.update(non_target_kwargs)
        return target_kwargs

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'transform': self.transform,
                'num_views': self.num_views
            })