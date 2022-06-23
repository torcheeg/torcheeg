from typing import Dict, List, Union

from ..base_transform import LabelTransform


class Select(LabelTransform):
    r'''
    Select part of the value from the information dictionary.

    .. code-block:: python

        transform = Select(key='valence')
        transform(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y']
        >>> 4.5

    :obj:`Select` allows multiple values to be selected and returned as a list. Suitable for multi-classification tasks or multi-task learning.

    .. code-block:: python

        transform = Select(key=['valence', 'arousal'])
        transform(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y']
        >>> [4.5, 5.5]

    Args:
        key (str or list): The selected key can be a key string or a list of keys.

    .. automethod:: __call__
    '''
    def __init__(self, key: Union[str, List]):
        super(Select, self).__init__()
        self.key = key
        self.select_list = isinstance(key, list) or isinstance(key, tuple)

    def __call__(self, *args, y: Dict, **kwargs) -> Union[int, float, List]:
        r'''
        Args:
            y (dict): A dictionary describing the EEG signal samples, usually as the last return value for each sample in :obj:`Dataset`.
            
        Returns:
            str or list: Selected value or selected value list.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Dict, **kwargs) -> Union[int, float, List]:
        assert isinstance(
            y, dict
        ), f'The transform Select only accepts label dict as input, but obtain {type(y)} as input.'
        if self.select_list:
            return [y[k] for k in self.key]
        return y[self.key]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'key': self.key
        })