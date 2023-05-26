from typing import Dict, Union

import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.fft import fft, ifft
from torch.nn.functional import pad

from ..base_transform import EEGTransform


class RandomEEGTransform(EEGTransform):
    def __init__(self, p: float = 0.5, apply_to_baseline: bool = False):
        super(RandomEEGTransform,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.p = p

    def apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.p < torch.rand(1):
            return eeg
        return self.random_apply(eeg, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'p': self.p})


class RandomNoise(RandomEEGTransform):
    '''
    Add random noise conforming to the normal distribution on the EEG signal.
    
    .. code-block:: python

        transform = RandomNoise(p=0.5)
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        mean (float): The mean of the normal distribution of noise. (default: :obj:`0.0`)
        std (float): The standard deviation of the normal distribution of noise. (default: :obj:`0.0`)
        p (float): Probability of adding noise to EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no noise is added to every sample and 1.0 means that noise is added to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 mean: float = 0.0,
                 std: float = 1.0,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomNoise, self).__init__(p=p,
                                          apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std

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

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        noise = torch.randn_like(eeg)
        noise = (noise + self.mean) * self.std
        return eeg + noise

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'mean': self.mean, 'std': self.std})


class RandomMask(RandomEEGTransform):
    '''
    Overlay the EEG signal using a random mask, and the value of the overlaid data points was set to 0.0.
    
    .. code-block:: python

        transform = RandomMask()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        ratio (float): The proportion of data points covered by the mask out of all data points for each EEG signal sample. (default: :obj:`0.5`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 ratio: float = 0.5,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomMask, self).__init__(p=p,
                                         apply_to_baseline=apply_to_baseline)
        self.ratio = ratio

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

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = torch.rand_like(eeg)
        mask = (mask < self.ratio).to(eeg.dtype)
        return eeg * mask

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'ratio': self.ratio})


class RandomWindowSlice(RandomEEGTransform):
    '''
    Randomly applies a slice transformation with a given probability, where the original time series is sliced by a window, and the sliced data is scaled to the original size. It is worth noting that the random position where each channel slice starts is the same.
    
    .. code-block:: python

        transform = RandomWindowSlice()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomWindowSlice(window_size=100)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomWindowSlice(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        window_size (int): The window size of the slice, the original signal will be sliced to the window_size size, and then adaptively scaled to the input shape.
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 window_size: int = 120,
                 series_dim: int = -1,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomWindowSlice,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.window_size = window_size
        self.series_dim = series_dim

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
            torch.Tensor: The output EEG signal after applying a random window slicing.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        eeg = eeg.numpy()

        assert len(eeg.shape) == 2 or len(
            eeg.shape
        ) == 3, 'Window slicing is only supported on 2D arrays or 3D arrays. In 2D arrays, the last dimension represents time series. In 3D arrays, the second dimension represents time series.'

        if self.window_size >= eeg.shape[self.series_dim]:
            return eeg

        assert -len(eeg.shape) <= self.series_dim < len(
            eeg.shape
        ), f'series_dim should be in range of [{- len(eeg.shape)}, {len(eeg.shape)}).'

        if self.series_dim < 0:
            self.series_dim = len(eeg.shape) + self.series_dim

        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            eeg = eeg.transpose(transpose_dims)

        if len(eeg.shape) == 2:
            starts = np.random.randint(low=0,
                                       high=eeg.shape[-1] - self.window_size,
                                       size=(eeg.shape[0])).astype(int)
        else:
            starts = np.random.randint(low=0,
                                       high=eeg.shape[-1] - self.window_size,
                                       size=(eeg.shape[0],
                                             eeg.shape[1])).astype(int)
        ends = (self.window_size + starts).astype(int)

        new_eeg = np.zeros_like(eeg)
        for i, eeg_i in enumerate(eeg):
            if len(eeg.shape) == 3:
                for j, eeg_i_j in enumerate(eeg_i):
                    new_eeg[i][j] = np.interp(
                        np.linspace(0, self.window_size, num=eeg.shape[-1]),
                        np.arange(self.window_size),
                        eeg_i_j[starts[i][j]:ends[i][j]]).T
            else:
                new_eeg[i] = np.interp(
                    np.linspace(0, self.window_size, num=eeg.shape[-1]),
                    np.arange(self.window_size), eeg_i[starts[i]:ends[i]]).T

        if self.series_dim != (len(eeg.shape) - 1):
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
            new_eeg = new_eeg.transpose(undo_transpose_dims)

        return torch.from_numpy(new_eeg)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'window_size': self.window_size,
                'series_dim': self.series_dim
            })


class RandomWindowWarp(RandomEEGTransform):
    '''
    Apply the window warping with a given probability, where a part of time series data is warpped by speeding it up or down.
    
    .. code-block:: python

        transform = RandomWindowWarp()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomWindowWarp(window_size=24, warp_size=48)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomWindowWarp(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        window_size (int): Randomly pick a window of size window_size on the time series to transform. (default: :obj:`-1`)
        warp_size (int): The size of the window after the warp. If warp_size is larger than window_size, it means slowing down, and if warp_size is smaller than window_size, it means speeding up. (default: :obj:`24`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 window_size: int = 12,
                 warp_size: int = 24,
                 series_dim: int = -1,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomWindowWarp,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.window_size = window_size
        self.warp_size = warp_size
        self.series_dim = series_dim

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
            torch.Tensor: The output EEG signal after applying a random window warping.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        eeg = eeg.numpy()

        assert len(eeg.shape) == 2 or len(
            eeg.shape
        ) == 3, 'Window warping is only supported on 2D arrays or 3D arrays. In 2D arrays, the last dimension represents time series. In 3D arrays, the second dimension represents time series.'

        if self.window_size >= eeg.shape[self.series_dim]:
            return eeg

        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            eeg = eeg.transpose(transpose_dims)

        window_steps = np.arange(self.window_size)
        if len(eeg.shape) == 2:
            starts = np.random.randint(low=0,
                                       high=eeg.shape[-1] - self.window_size,
                                       size=(eeg.shape[0])).astype(int)
        else:
            starts = np.random.randint(low=0,
                                       high=eeg.shape[-1] - self.window_size,
                                       size=(eeg.shape[0],
                                             eeg.shape[1])).astype(int)
        ends = (self.window_size + starts).astype(int)

        new_eeg = np.zeros_like(eeg)
        for i, eeg_i in enumerate(eeg):
            if len(eeg.shape) == 3:
                for j, eeg_i_j in enumerate(eeg_i):
                    start_seg = eeg_i_j[:starts[i][j]]
                    window_seg = np.interp(
                        np.linspace(0, self.window_size - 1,
                                    num=self.warp_size), window_steps,
                        eeg_i_j[starts[i][j]:ends[i][j]])
                    end_seg = eeg_i_j[ends[i][j]:]
                    warped = np.concatenate((start_seg, window_seg, end_seg))
                    new_eeg[i][j] = np.interp(
                        np.arange(eeg.shape[-1]),
                        np.linspace(0, eeg.shape[-1] - 1., num=warped.size),
                        warped).T
            else:
                start_seg = eeg_i[:starts[i]]
                window_seg = np.interp(
                    np.linspace(0, self.window_size - 1, num=self.warp_size),
                    window_steps, eeg_i[starts[i]:ends[i]])
                end_seg = eeg_i[ends[i]:]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                new_eeg[i] = np.interp(
                    np.arange(eeg.shape[-1]),
                    np.linspace(0, eeg.shape[-1] - 1., num=warped.size),
                    warped).T

        if self.series_dim != (len(eeg.shape) - 1):
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
            new_eeg = new_eeg.transpose(undo_transpose_dims)

        return torch.from_numpy(new_eeg)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'window_size': self.window_size,
                'warp_size': self.warp_size,
                'series_dim': self.series_dim
            })


class RandomPCANoise(RandomEEGTransform):
    '''
    Add noise with a given probability, where the noise is added to the principal components of each channel of the EEG signal. In particular, the noise added by each channel is different.
    
    .. code-block:: python

        transform = RandomPCANoise()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomPCANoise(mean=0.5, std=2.0, n_components=4)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomPCANoise(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        mean (float): The mean of the normal distribution of noise. (default: :obj:`0.0`)
        std (float): The standard deviation of the normal distribution of noise. (default: :obj:`0.0`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        n_components (int): Number of components to add noise. if n_components is not set, the first two components are used to add noise.
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 mean: float = 0.0,
                 std: float = 1.0,
                 n_components: int = 2,
                 series_dim: int = -1,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomPCANoise,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std
        self.n_components = n_components
        self.series_dim = series_dim

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
            torch.Tensor: The output EEG signal after applying a random PCA noise.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        eeg = eeg.numpy()

        assert len(eeg.shape) == 2 or len(
            eeg.shape
        ) == 3, 'Window warping is only supported on 2D arrays or 3D arrays. In 2D arrays, the last dimension represents time series. In 3D arrays, the second dimension represents time series.'

        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            eeg = eeg.transpose(transpose_dims)

        pca = PCA(n_components=self.n_components)
        pca.fit(eeg.reshape(-1, eeg.shape[-1]))
        components = pca.components_
        variances = pca.explained_variance_ratio_

        new_eeg = np.zeros_like(eeg)
        for i, eeg_i in enumerate(eeg):
            if len(eeg.shape) == 3:
                for j, eeg_i_j in enumerate(eeg_i):
                    coeffs = np.random.normal(loc=self.mean,
                                              scale=self.std,
                                              size=pca.n_components) * variances
                    new_eeg[i][j] = eeg_i_j + (components * coeffs.reshape(
                        (pca.n_components, -1))).sum(axis=0)
            else:
                coeffs = np.random.normal(loc=self.mean,
                                          scale=self.std,
                                          size=pca.n_components) * variances
                new_eeg[i] = eeg_i + (components * coeffs.reshape(
                    (pca.n_components, -1))).sum(axis=0)

        if self.series_dim != (len(eeg.shape) - 1):
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
            new_eeg = new_eeg.transpose(undo_transpose_dims)

        return torch.from_numpy(new_eeg)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'mean': self.mean,
                'std': self.std,
                'n_components': self.n_components,
                'series_dim': self.series_dim
            })


class RandomFlip(RandomEEGTransform):
    '''
    Applies a random transformation with a given probability to reverse the direction of the input signal in the specified dimension, commonly used for left-right and bottom-up reversal of EEG caps and reversal of timing.
    
    .. code-block:: python

        transform = RandomFlip(dim=-1)
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomFlip(dim=1)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        dim (int): Dimension to be flipped in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self, dim=-1, p: float = 0.5, apply_to_baseline: bool = False):
        super(RandomFlip, self).__init__(p=p,
                                         apply_to_baseline=apply_to_baseline)
        self.dim = dim

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
            torch.Tensor: The output EEG signal after applying a random flipping.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.flip(eeg, dims=(self.dim, ))

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'dim': self.dim})


class RandomSignFlip(RandomEEGTransform):
    '''
    Apply a random transformation such that the input signal becomes the opposite of the reversed sign with a given probability
    
    .. code-block:: python

        transform = RandomSignFlip()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self, p: float = 0.5, apply_to_baseline: bool = False):
        super(RandomSignFlip,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)

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
            torch.Tensor: The output EEG signal after applying a random sign flipping.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        return -eeg


class RandomShift(RandomEEGTransform):
    '''
    Apply a shift with a specified probability, after which the specified dimension is shifted backward, and the part shifted out of the Tensor is added to the front of that dimension.
    
    .. code-block:: python

        transform = RandomShift(dim=-1, shift_min=8, shift_max=24)
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        shift_min (float or int): The minimum shift in the random transformation. (default: :obj:`-2.0`)
        shift_max (float or int): The maximum shift in random transformation. (default: :obj:`2.0`)
        dim (int): Dimension to be shifted in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 p: float = 0.5,
                 shift_min: int = 8,
                 shift_max: int = 12,
                 dim: int = -1,
                 apply_to_baseline: bool = False):
        super(RandomShift, self).__init__(p=p,
                                          apply_to_baseline=apply_to_baseline)
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.dim = dim

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
            torch.Tensor: The output EEG signal after applying a random shift.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        shift = torch.randint(low=self.shift_min,
                              high=self.shift_max,
                              size=(1, ))
        return torch.roll(eeg, shifts=shift.item(), dims=self.dim)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'shift_min': self.shift_min,
                'shift_max': self.shift_max,
                'dim': self.dim
            })


class RandomChannelShuffle(RandomEEGTransform):
    '''
    Apply a shuffle with a specified probability, after which the order of the channels is randomly shuffled.
    
    .. code-block:: python

        transform = RandomChannelShuffle()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self, p: float = 0.5, apply_to_baseline: bool = False):
        super(RandomChannelShuffle,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)

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
            torch.Tensor: The output EEG signal after applying a random channel shuffle.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        index_list = np.arange(len(eeg))
        np.random.shuffle(index_list)
        return eeg[index_list]


class RandomHemisphereChannelShuffle(RandomEEGTransform):
    '''
    Apply a shuffle with a specified probability on a single hemisphere (either left or right), after which the order of the channels is randomly shuffled.
    
    .. code-block:: python

        transform = RandomChannelShuffle(location_list=M3CV_LOCATION_LIST,
                                         channel_location_dict=M3CV_CHANNEL_LOCATION_DICT)
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

    Args:
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 location_list,
                 channel_location_dict,
                 p: float = 0.5,
                 apply_to_baseline: bool = False):
        super(RandomHemisphereChannelShuffle,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.location_list = location_list
        self.channel_location_dict = channel_location_dict

        width = len(location_list[0])
        left_channel_list = []
        right_channel_list = []
        for i, (loc_y, loc_x) in enumerate(channel_location_dict.values()):
            if loc_x < width // 2:
                left_channel_list.append(i)
            if loc_y > width // 2:
                right_channel_list.append(i)
        self.left_channel_list = left_channel_list
        self.right_channel_list = right_channel_list

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
            torch.Tensor: The output EEG signal after applying a random channel shuffle.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        if 0.5 < torch.rand(1):
            index_list = self.left_channel_list
        else:
            index_list = self.right_channel_list

        shuffle_index_list = np.random.permutation(index_list.copy())
        eeg[index_list] = eeg[shuffle_index_list]
        return eeg

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'location_list': [...],
            'channel_location_dict': {...}
        })


class RandomFrequencyShift(RandomEEGTransform):
    '''
    Apply a frequency shift with a specified probability, after which the EEG signals of all channels are equally shifted in the frequency domain.
    
    .. code-block:: python

        transform = RandomFrequencyShift()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomFrequencyShift(sampling_rate=128, shift_min=4.0)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomFrequencyShift(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        sampling_rate (int): The original sampling rate in Hz (default: :obj:`128`)
        shift_min (float or int): The minimum shift in the random transformation. (default: :obj:`-2.0`)
        shift_max (float or int): The maximum shift in random transformation. (default: :obj:`2.0`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 p: float = 0.5,
                 sampling_rate: int = 128,
                 shift_min: Union[float, int] = -2.0,
                 shift_max: Union[float, int] = 2.0,
                 series_dim: int = 0,
                 apply_to_baseline: bool = False):
        super(RandomFrequencyShift,
              self).__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.series_dim = series_dim

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
            torch.Tensor: The output EEG signal after applying a random sampling_rate shift.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.series_dim != (len(eeg.shape) - 1):
            permute_dims = list(range(len(eeg.shape)))
            permute_dims.pop(self.series_dim)
            permute_dims = [*permute_dims, self.series_dim]
            eeg = eeg.permute(permute_dims)

        N_orig = eeg.shape[-1]
        N_padded = 2**int(np.ceil(np.log2(np.abs(N_orig))))
        t = torch.arange(N_padded) / self.sampling_rate
        padded = pad(eeg, (0, N_padded - N_orig))

        if torch.is_complex(eeg):
            raise ValueError("eeg must be real.")

        N = padded.shape[-1]
        f = fft(padded, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2

        analytical = ifft(f * h, dim=-1)

        shift = torch.rand(1) * (self.shift_max -
                                 self.shift_min) + self.shift_min
        shifted = analytical * torch.exp(2j * np.pi * shift * t)

        shifted = shifted[..., :N_orig].real.float()

        if self.series_dim != (len(eeg.shape) - 1):
            undo_permute_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(permute_dims):
                undo_permute_dims[dim] = i
            shifted = shifted.permute(undo_permute_dims)

        return shifted

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sampling_rate': self.sampling_rate,
                'shift_min': self.shift_min,
                'shift_max': self.shift_max,
                'series_dim': self.series_dim
            })
