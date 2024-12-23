from typing import Dict, List, Union, Optional

from ..base_transform import LabelTransform


class Normalize(LabelTransform):
    r'''
    Normalize the label using min-max normalization or standardization.

    For min-max normalization:
    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Normalize(min=0.0, max=1.0)
        t(y=0.5)['y']
        >>> 0.5

    For standardization:
    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Normalize(mean=0.0, std=1.0)
        t(y=0.5)['y']
        >>> 0.5

    Args:
        min (float, optional): Minimum value for min-max normalization. Default: None
        max (float, optional): Maximum value for min-max normalization. Default: None
        mean (float, optional): Mean value for standardization. Default: None
        std (float, optional): Standard deviation value for standardization. Default: None

    Note:
        Either (min, max) or (mean, std) should be provided, but not both.

    .. automethod:: __call__
    '''

    def __init__(self,
                 min: Optional[float] = None,
                 max: Optional[float] = None,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):
        super(Normalize, self).__init__()

        if (min is not None and max is not None) and (mean is None and std is None):
            self.mode = 'minmax'
            self.min = min
            self.max = max
        elif (mean is not None and std is not None) and (min is None and max is None):
            self.mode = 'standard'
            self.mean = mean
            self.std = std
        else:
            raise ValueError(
                'Either (min, max) or (mean, std) should be provided, but not both.')

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[float, List]:
        if isinstance(y, list):
            if self.mode == 'minmax':
                return [(float(l) - self.min) / (self.max - self.min) for l in y]
            else:
                return [(float(l) - self.mean) / self.std for l in y]
        else:
            if self.mode == 'minmax':
                return (float(y) - self.min) / (self.max - self.min)
            else:
                return (float(y) - self.mean) / self.std

    @property
    def repr_body(self) -> Dict:
        if self.mode == 'minmax':
            return dict(super().repr_body, **{
                'min': self.min,
                'max': self.max
            })
        else:
            return dict(super().repr_body, **{
                'mean': self.mean,
                'std': self.std
            })
