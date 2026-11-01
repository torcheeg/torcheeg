from ..base_transform import EEGTransform


class BaselineRemoval(EEGTransform):
    """Subtract a baseline with the same shape as the input signal."""

    def __init__(self, baseline):
        self.baseline = baseline

    def apply(self, eeg):
        if self.baseline.shape != eeg.shape:
            raise ValueError('baseline and eeg must have the same shape.')
        return eeg - self.baseline


class BaselineCorrection(EEGTransform):
    """Subtract the baseline mean along axis from the input signal."""

    def __init__(self, baseline, axis=-1):
        self.baseline = baseline
        self.axis = axis

    def apply(self, eeg):
        return eeg - self.baseline.mean(self.axis, keepdims=True)
