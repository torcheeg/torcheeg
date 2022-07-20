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
        transforms (list, tuple): a sequence of transforms
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transforms: Sequence[Callable],
                 apply_to_baseline: bool = False):
        super(Concatenate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.transforms = transforms

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
            np.ndarray: The combined results of multiple transforms.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        out = []
        for t in self.transforms:
            out.append(t.apply(eeg, **kwargs))
        return np.concatenate(out, axis=-1)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ChunkConcatenate(EEGTransform):
    r'''
    Divide the input EEG signal into multiple chunks according to chunk_size and overlap, and then apply transforms to each chunk, and combine the calculation results of all transforms on all chunks. It is used when feature fusion is required.

    .. code-block:: python

        transform = ChunkConcatenate([
            BandDifferentialEntropy(),
            BandMeanAbsoluteDeviation()
        ],
        chunk_size=250,
        overlap=0)
        transform(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 32)

    TorchEEG allows feature fusion at multiple scales:

    .. code-block:: python

        transform = Concatenate([
            ChunkConcatenate([
                BandDifferentialEntropy()
            ],
            chunk_size=250,
            overlap=0),  # 4 chunk * 4-dim feature
            ChunkConcatenate([
                BandDifferentialEntropy()
            ],
            chunk_size=500,
            overlap=0),  # 2 chunk * 4-dim feature
            BandDifferentialEntropy()  # 1 chunk * 4-dim feature
        ])
        transform(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 28) # 4 * 4 + 2 * 4 + 1 * 4

    Args:
        transforms (list, tuple): a sequence of transforms
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transforms: Sequence[Callable],
                 chunk_size: int = 250,
                 overlap: int = 0,
                 apply_to_baseline: bool = False):
        super(ChunkConcatenate,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.transforms = transforms
        self.chunk_size = chunk_size
        self.overlap = overlap

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
            np.ndarray: The combined results of multiple transforms.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        start_at = 0
        end_at = start_at + self.chunk_size
        step = self.chunk_size - self.overlap

        out = []

        while end_at <= eeg.shape[1]:

            for t in self.transforms:
                out.append(t.apply(eeg[:, start_at:end_at], **kwargs))

            start_at = start_at + step
            end_at = start_at + self.chunk_size

        return np.concatenate(out, axis=-1)

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
