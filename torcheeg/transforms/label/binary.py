from typing import Dict, List, Union

import numpy as np

from ..base_transform import LabelTransform


class Binary(LabelTransform):
    r'''
    Binarize the label according to a certain threshold. Labels larger than the threshold are set to 1, and labels smaller than the threshold are set to 0.
    
    .. code-block:: python

        transform = Binary(threshold=5.0)
        transform(y=4.5)['y']
        >>> 0

    :obj:`Binary` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        transform = Binary(threshold=5.0)
        transform(y=[4.5, 5.5])['y']
        >>> [0, 1]

    Args:
        threshold (float): Threshold used during binarization.

    .. automethod:: __call__
    '''
    def __init__(self, threshold: float):
        super(Binary, self).__init__()
        self.threshold = threshold

    def __call__(self, *args, y: Union[int, float, List],
                 **kwargs) -> Union[int, List]:
        r'''
        Args:
            label (int, float, or list): The input label or list of labels.
            
        Returns:
            int, float, or list: The output label or list of labels after binarization.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        if isinstance(y, list):
            return [int(l >= self.threshold) for l in y]
        return int(y >= self.threshold)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'threshold': self.threshold})


class BinaryOneVSRest(LabelTransform):
    r'''
    Binarize the label following the fashion of the one-vs-rest strategy. When label is the specified positive category label, the label is set to 1, when the label is any other category label, the label is set to 0.
    
    .. code-block:: python

        transform = BinaryOneVSRest(positive=1)
        transform(y=2)['y']
        >>> 0

    :obj:`Binary` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        transform = BinaryOneVSRest(positive=1)
        transform(y=[1, 2])['y']
        >>> [1, 0]

    Args:
        positive (int): The specified positive category label.

    .. automethod:: __call__
    '''
    def __init__(self, positive: int):
        super(BinaryOneVSRest, self).__init__()
        self.positive = positive

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        assert isinstance(
            y, (int, float, list)
        ), f'The transform Binary only accepts label list or item (int or float) as input, but obtain {type(y)} as input.'
        if isinstance(y, list):
            return [int(l == self.positive) for l in y]
        return int(y == self.positive)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'positive': self.positive})


class BinariesToCategory(LabelTransform):
    r'''
    Convert multiple binary labels into one multiclass label. Multiclass labels represent permutations of binary labels. Commonly used to combine two binary classification tasks into a single quad classification task.
    
    .. code-block:: python

        transform = BinariesToCategory()
        transform(y=[0, 0])['y']
        >>> 0
        transform(y=[0, 1])['y']
        >>> 1
        transform(y=[1, 0])['y']
        >>> 2
        transform(y=[1, 1])['y']
        >>> 3
    
    .. automethod:: __call__
    '''
    def __call__(self, *args, y: List, **kwargs) -> int:
        r'''
        Args:
            y (list): list of binary labels.
            
        Returns:
            int: The converted multiclass label.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: List, **kwargs) -> int:
        r'''
        Args:
            y (list): list of binary labels.
            
        Returns:
            int: The converted multiclass label.
        '''
        assert isinstance(
            y, list
        ), f'The transform BinariesToCategory only accepts label list as input, but obtain {type(y)} as input.'
        return sum([v * 2**i for i, v in enumerate(reversed(y))])
