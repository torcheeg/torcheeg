from typing import Dict, List, Union

from ..base_transform import LabelTransform


class FixCategory(LabelTransform):
    r"""
    Returns a pre-set label for all samples, usually used to supplement the dataset with new categories.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.FixCategory(value=0)
        t(3)
        >>> 0

    :obj:`FixCategory` allows multiple values to be selected and returned as a list. Suitable for multi-classification tasks or multi-task learning.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.FixCategory(value=[0, 1])
        t([1, 2])
        >>> [0, 1]

    Args:
        value (str or list): The pre-set label.

    .. automethod:: __call__
    """

    def __init__(self, value: Union[int, str, List]):
        super(FixCategory, self).__init__()
        self.value = value

    def apply(self, y: Dict, **kwargs) -> Union[int, float, List]:
        return self.value

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'value': self.value
        })
