from typing import TypeVar

import numpy as np
import torch

from ..base_transform import EEGTransform

_Signal = TypeVar("_Signal", np.ndarray, torch.Tensor)


class BaselineRemoval(EEGTransform):
    """Subtract a baseline with the same shape as the input signal."""

    def __init__(self, baseline: _Signal):
        self.baseline = baseline

    def __call__(self, eeg: _Signal) -> _Signal:
        if self.baseline.shape != eeg.shape:
            raise ValueError('baseline and eeg must have the same shape.')
        return eeg - self.baseline


class BaselineCorrection(EEGTransform):
    """Subtract the baseline mean along axis from the input signal."""

    def __init__(self, baseline: _Signal, axis: int = -1):
        self.baseline = baseline
        self.axis = axis

    def __call__(self, eeg: _Signal) -> _Signal:
        return eeg - self.baseline.mean(self.axis, keepdims=True)
