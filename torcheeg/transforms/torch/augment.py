import torch
from typing import Union, Sequence

from torch.nn.functional import interpolate


class Resize:
    r'''
    Use an interpolation algorithm to scale a grid-like EEG signal at the spatial dimension.

    .. code-block:: python

        transform = ToTensor(size=(64, 64))
        transform(torch.randn(128, 9, 9)).shape
        >>> (128, 64, 64)

    Args:
        size (tuple): The output spatial size.
        interpolation (str): The interpolation algorithm used for upsampling, can be nearest, linear, bilinear, bicubic, trilinear, and area. (defualt: :obj:`'nearest'`)

    .. automethod:: __call__
    '''
    def __init__(
        self,
        size: Union[Sequence[int], int],
        interpolation: str = "bilinear",
    ):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): The input EEG signal in shape of [height of grid, width of grid, number of data points].

        Returns:
            torch.Tensor[new height of grid, new width of grid, number of subbands]: The scaled EEG signal at the saptial dimension.
        '''
        assert x.dim() == 3, f'The module only allows to input a 3-d tensor, but the input has dimension {x.dim()}'

        x = x.unsqueeze(0)

        align_corners = False if self.interpolation in ["bilinear", "bicubic"] else None

        interpolated_x = interpolate(x, size=self.size, mode=self.interpolation, align_corners=align_corners)

        return interpolated_x.squeeze(0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomNoise:
    '''
    Add random noise conforming to the normal distribution on the EEG signal.
    
    .. code-block:: python

        transform = RandomNoise(p=0.5)
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

    Args:
        mean (float): The mean of the normal distribution of noise. (defualt: :obj:`0.0`)
        std (float): The standard deviation of the normal distribution of noise. (defualt: :obj:`0.0`)
        p (float): Probability of adding noise to EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no noise is added to every sample and 1.0 means that noise is added to every sample. (defualt: :obj:`0.5`)

    .. automethod:: __call__
    '''
    def __init__(self, mean: float = 0.0, std: float = 1.0, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): The input EEG signal.

        Returns:
            torch.Tensor: The output EEG signal after adding random noise.
        '''
        if self.p < torch.rand(1):
            return x
        noise = torch.randn_like(x)
        noise = (noise + self.mean) * self.std
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomMask:
    '''
    Overlay the EEG signal using a random mask, and the value of the overlaid data points was set to 0.0.
    
    .. code-block:: python

        transform = RandomMask()
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

    Args:
        ratio (float): The proportion of data points covered by the mask out of all data points for each EEG signal sample. (defualt: :obj:`0.5`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (defualt: :obj:`0.5`)
    .. automethod:: __call__
    '''
    def __init__(self, ratio: float = 0.5, p: float = 0.5):
        self.ratio = ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): The input EEG signal.

        Returns:
            torch.Tensor: The output EEG signal after applying a random mask.
        '''
        if self.p < torch.rand(1):
            return x
        mask = torch.rand_like(x)
        mask = (mask < self.ratio).to(x.dtype)
        return x * mask

    def __repr__(self):
        return f"{self.__class__.__name__}()"