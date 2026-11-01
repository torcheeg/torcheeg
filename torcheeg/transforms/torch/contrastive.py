from copy import deepcopy

from ..base_transform import EEGTransform


class Contrastive(EEGTransform):
    """Generate independent transformed views of one signal."""

    def __init__(self, transform, num_views=2):
        if num_views < 1:
            raise ValueError('num_views must be positive.')
        self.transform = transform
        self.num_views = num_views

    def __call__(self, eeg):
        return [self.transform(deepcopy(eeg)) for _ in range(self.num_views)]

    @property
    def repr_body(self):
        return {'transform': self.transform, 'num_views': self.num_views}
