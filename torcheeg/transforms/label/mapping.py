from typing import Dict, List, Union

from ..base_transform import LabelTransform


class Mapping(LabelTransform):
    r'''
    Mapping the label according to a certain dictionary.
    
    .. code-block:: python

        transform = Mapping({
            'left_hand': 0,
            'right_hand': 1,
        })
        transform(y='left_hand')['y']
        >>> 0

    :obj:`Mapping` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        transform = Mapping({
            'left_hand': 0,
            'right_hand': 1,
            'left_feet': 0,
            'right_feet': 1
        })
        transform(y=['left_hand', 'left_feet'])['y']
        >>> [0, 0]

    Args:
        map_dict (float): The mapping dictionary.
        default (float, optional): The default value when the input label is not in the dictionary. (default: :obj:`-1`)

    .. automethod:: __call__
    '''
    def __init__(self, map_dict: float, default: float = -1) -> None:
        super(Mapping, self).__init__()
        self.map_dict = map_dict
        self.default = default

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
            return [self.map_dict.get(label, self.default) for label in y]
        else:
            return self.map_dict.get(y, self.default)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'map_dict': self.map_dict})