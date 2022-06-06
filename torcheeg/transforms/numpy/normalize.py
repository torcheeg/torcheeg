import numpy as np

from typing import Union


class MeanStdNormalize:
    r'''
    Perform z-score normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        transform = Concatenate([
            MeanStdNormalize(axis=0)
        ])
        # normalize along the first dimension (electrode dimension)
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

        transform = Concatenate([
            MeanStdNormalize(axis=1)
        ])
        # normalize along the second dimension (temproal dimension)
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

    Args:
        mean (np.array, optional): The mean used in the normalization process, allowing the user to provide mean statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        std (np.array, optional): The standard deviation used in the normalization process, allowing the user to provide tandard deviation statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
    
    .. automethod:: __call__
    '''
    def __init__(self, mean: Union[np.ndarray, None] = None, std: Union[np.ndarray, None] = None, axis: Union[int, None] = None):
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(self, x: np.ndarray):
        r'''
        Args:
            x (np.ndarray): The input EEG signals or features.

        Returns:
            np.ndarray: The normalized results.
        '''
        if (self.mean is None) or (self.std is None):
            if self.axis is None:
                mean = x.mean()
                std = x.std()
            else:
                mean = x.mean(axis=self.axis, keepdims=True)
                std = x.std(axis=self.axis, keepdims=True)
        elif not self.axis is None:
            shape = [-1] * len(x.shape)
            shape[self.axis] = 1
            mean = self.mean.reshape(*shape)
            std = self.std.reshape(*shape)
        return (x - mean) / std

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MinMaxNormalize:
    r'''
    Perform min-max normalization on the input data. This class allows the user to define the dimension of normalization and the used statistic.

    .. code-block:: python

        transform = Concatenate([
            MinMaxNormalize(axis=0)
        ])
        # normalize along the first dimension (electrode dimension)
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

        transform = Concatenate([
            MinMaxNormalize(axis=1)
        ])
        # normalize along the second dimension (temproal dimension)
        transform(torch.randn(32, 128)).shape
        >>> (32, 128)

    Args:
        min (np.array, optional): The minimum used in the normalization process, allowing the user to provide minimum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        max (np.array, optional): The maximum used in the normalization process, allowing the user to provide maximum statistics in :obj:`np.ndarray` format. When statistics are not provided, use the statistics of the current sample for normalization.
        axis (int, optional): The dimension to normalize, when no dimension is specified, the entire data is normalized.
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 min: Union[np.ndarray, None, float] = None,
                 max: Union[np.ndarray, None, float] = None,
                 axis: Union[int, None] = None):
        self.min = min
        self.max = max
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals or features.
            
        Returns:
            np.ndarray: The normalized results.
        '''
        if (self.min is None) or (self.max is None):
            if self.axis is None:
                min = x.min()
                max = x.max()
            else:
                min = x.min(axis=self.axis, keepdims=True)
                max = x.max(axis=self.axis, keepdims=True)
        elif not self.axis is None:
            shape = [-1] * len(x.shape)
            shape[self.axis] = 1
            min = self.min.reshape(*shape)
            max = self.max.reshape(*shape)

        return (x - min) / (max - min)

    def __repr__(self):
        return f"{self.__class__.__name__}()"