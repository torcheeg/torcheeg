from typing import Dict, Sequence, Union

import torch
from torch.nn.functional import interpolate

from ..base_transform import EEGTransform


class Resize(EEGTransform):
    r'''
    Use an interpolation algorithm to scale a grid-like EEG signal at the spatial dimension.

    .. code-block:: python

        transform = ToTensor(size=(64, 64))
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 64, 64)

    Args:
        size (tuple): The output spatial size.
        interpolation (str): The interpolation algorithm used for upsampling, can be nearest, linear, bilinear, bicubic, trilinear, and area. (default: :obj:`'nearest'`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 size: Union[Sequence[int], int],
                 interpolation: str = "bilinear",
                 apply_to_baseline: bool = False):
        super(Resize, self).__init__(apply_to_baseline=apply_to_baseline)
        self.size = size
        self.interpolation = interpolation

    def __call__(self,
                 *args,
                 eeg: torch.Tensor,
                 baseline: Union[torch.Tensor, None] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        r'''
        Args:
            eeg (torch.Tensor): The input EEG signal in shape of [height of grid, width of grid, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch.Tensor[new height of grid, new width of grid, number of sub-bands]: The scaled EEG signal at the saptial dimension.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        assert eeg.dim() == 3, f'The Resize only allows to input a 3-d tensor, but the input has dimension {eeg.dim()}'

        eeg = eeg.unsqueeze(0)

        align_corners = False if self.interpolation in ["bilinear", "bicubic"] else None

        interpolated_x = interpolate(eeg, size=self.size, mode=self.interpolation, align_corners=align_corners)

        return interpolated_x.squeeze(0)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'size': self.size, 'interpolation': self.interpolation})