from typing import Callable, Sequence, Dict, Union

import numpy as np

from ..base_transform import EEGTransform


class Concatenate(EEGTransform):
    r'''
    Merge the calculation results of multiple transforms, which are used when feature fusion is required.

    .. code-block:: python

        transform = Concatenate([
            BandDifferentialEntropy(),
            BandMeanAbsoluteDeviation()
        ])
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 8)

    Args:
        transforms (list, tuple): a sequence of transforms.
        axis (int): The axis along which the arrays will be joined. If axis is None, arrays are flattened before use (default: :obj:`-1`).
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transforms: Sequence[Callable],
                 axis: int = -1,
                 apply_to_baseline: bool = False):
        super(Concatenate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.transforms = transforms
        self.axis = axis

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            np.ndarray: The combined results of multiple transforms.
        '''
        if args:
            raise KeyError("Please pass data as named parameters.")

        target_kwargs = {}
        non_target_kwargs = {}

        for t in self.transforms:
            new_kwargs_t = t(**kwargs)
            for new_kwargs_key, new_kwargs_value in new_kwargs_t.items():
                if not new_kwargs_key in self.targets:
                    non_target_kwargs[new_kwargs_key] = new_kwargs_value
                    continue
                assert isinstance(
                    new_kwargs_value, np.ndarray
                ), f'Concate only supports concatenating numpy.ndarray type data, you are trying to concatenate {type(new_kwargs_value)} type data.'
                if not new_kwargs_key in target_kwargs:
                    target_kwargs[new_kwargs_key] = []
                target_kwargs[new_kwargs_key].append(new_kwargs_value)

        for target_kwargs_key, target_kwargs_value in target_kwargs.items():
            target_kwargs[target_kwargs_key] = np.concatenate(
                target_kwargs_value, axis=self.axis)

        target_kwargs.update(non_target_kwargs)
        return target_kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class MapChunk(EEGTransform):
    r'''
    Divide the input EEG signal into multiple chunks according to chunk_size and overlap, and then apply a transofrm to each chunk, and combine the calculation results of a transofrm on all chunks. It is used when feature fusion is required.

    .. code-block:: python

        transform = MapChunk(
            BandDifferentialEntropy(),
            chunk_size=250,
            overlap=0
        )
        transform(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 16)

    TorchEEG allows feature fusion at multiple scales:

    .. code-block:: python

        transform = Concatenate([
            MapChunk(
                BandDifferentialEntropy()
                chunk_size=250,
                overlap=0),  # 4 chunk * 4-dim feature
            MapChunk(
                BandDifferentialEntropy()
                chunk_size=500,
                overlap=0),  # 2 chunk * 4-dim feature
            BandDifferentialEntropy()  # 1 chunk * 4-dim feature
        ])
        transform(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 28) # 4 * 4 + 2 * 4 + 1 * 4

    Args:
        transform (EEGTransform): a transform
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transform: EEGTransform,
                 chunk_size: int = 250,
                 overlap: int = 0,
                 apply_to_baseline: bool = False):
        super(MapChunk,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.transform = transform
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            np.ndarray: The combined results of a transform from multiple chunks.
        '''
        target_kwargs = {}
        non_target_kwargs = {}

        check_len_key = None
        check_len_value = None

        for kwargs_key, kwargs_value in kwargs.items():
            if not kwargs_key in self.targets:
                non_target_kwargs[kwargs_key] = kwargs_value
                continue

            if not check_len_key:
                check_len_key = kwargs_key
                check_len_value = len(kwargs_value)
            else:
                assert len(
                    kwargs_value
                ) == check_len_value, f'The lengths of {check_len_key} ({check_len_value}) and {kwargs_key} ({len(kwargs_value)}) in the input signal are not the same.'

            start_at = 0
            end_at = start_at + self.chunk_size
            step = self.chunk_size - self.overlap

            chunk_kwargs_value = []
            while end_at <= kwargs_value.shape[1]:
                chunk_kwargs_value.append(kwargs_value[:, start_at:end_at])
                start_at = start_at + step
                end_at = start_at + self.chunk_size
            target_kwargs[kwargs_key] = chunk_kwargs_value

        new_target_kwargs = {}
        num_chunk = len(list(target_kwargs.values())[0])
        for chunk_index in range(num_chunk):
            cur_target_kwargs = {
                k: v[chunk_index]
                for k, v in target_kwargs.items()
            }
            cur_target_kwargs.update(non_target_kwargs)
            new_kwargs_t = self.transform(**kwargs)

            for new_kwargs_key, new_kwargs_value in new_kwargs_t.items():
                if not new_kwargs_key in self.targets:
                    continue
                assert isinstance(
                    new_kwargs_value, np.ndarray
                ), f'Concate only supports concatenating numpy.ndarray type data, you are trying to concatenate {type(new_kwargs_value)} type data.'
                if not new_kwargs_key in new_target_kwargs:
                    new_target_kwargs[new_kwargs_key] = []
                new_target_kwargs[new_kwargs_key].append(new_kwargs_value)

        for new_target_kwargs_key, new_target_kwargs_value in new_target_kwargs.items(
        ):
            new_target_kwargs[new_target_kwargs_key] = np.concatenate(
                new_target_kwargs_value, axis=-1)

        new_target_kwargs.update(non_target_kwargs)
        return new_target_kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(['
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n],'
        format_string += f'\nchunk_size={self.chunk_size}, '
        format_string += f'\noverlap={self.overlap})'
        return format_string
