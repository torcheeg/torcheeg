from copy import deepcopy

import numpy as np

from ..base_transform import EEGTransform


class Concatenate(EEGTransform):
    """Concatenate independent transform outputs along axis."""

    def __init__(self, transforms, axis=-1):
        self.transforms = tuple(transforms)
        self.axis = axis

    def apply(self, eeg):
        outputs = [transform(deepcopy(eeg)) for transform in self.transforms]
        if not all(isinstance(value, np.ndarray) for value in outputs):
            raise TypeError('Concatenate transforms must return numpy arrays.')
        return np.concatenate(outputs, axis=self.axis)

    @property
    def repr_body(self):
        return {'transforms': self.transforms, 'axis': self.axis}


class MapChunk(EEGTransform):
    """Transform complete windows and concatenate their features."""

    def __init__(self, transform, chunk_size=250, overlap=0):
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError(
                'Require chunk_size > 0 and 0 <= overlap < chunk_size.')
        self.transform = transform
        self.chunk_size = chunk_size
        self.overlap = overlap

    def apply(self, eeg):
        outputs = [self.transform(eeg[:, start:start + self.chunk_size].copy())
                   for start in range(0, eeg.shape[1] - self.chunk_size + 1,
                                      self.chunk_size - self.overlap)]
        if not outputs:
            raise ValueError('Input must contain at least one complete chunk.')
        if not all(isinstance(value, np.ndarray) for value in outputs):
            raise TypeError('MapChunk transforms must return numpy arrays.')
        return np.concatenate(outputs, axis=-1)

    @property
    def repr_body(self):
        return {'transform': self.transform, 'chunk_size': self.chunk_size,
                'overlap': self.overlap}
