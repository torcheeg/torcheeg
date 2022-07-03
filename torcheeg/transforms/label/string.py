import re

from typing import List, Union, Dict

from ..base_transform import LabelTransform


class StringToInt(LabelTransform):
    r'''
    Identify numbers in strings and convert strings to numbers. If there is no number in the string, the output corresponding to the string is 0.
    
    .. code-block:: python

        transform = StringToInt()
        transform(y='None')['y']
        >>> 0

        transform = StringToInt()
        transform(y='sub001')['y']
        >>> 1

    :obj:`StringToInt` allows converting a list of strings to a list of numbers with the same conversion behavior as a single string.

    .. code-block:: python

        transform = StringToInt()
        transform(y=['sub001', '4'])['y']
        >>> 1, 4

    .. automethod:: __call__
    '''
    def __init__(self):
        super(StringToInt, self).__init__()

    def __call__(self, *args, y: Union[str, List[str]], **kwargs) -> Union[int, List[int]]:
        r'''
        Args:
            label (str): The input label or list of labels.
            
        Returns:
            int: The output label or list of labels after binarization.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Union[str, List[str]], **kwargs) -> Union[int, List[int]]:
        if isinstance(y, list):
            return [self.opt(i) for i in y]
        return self.opt(y)

    def opt(self, y: str) -> int:
        if not isinstance(y, str):
            return y
        nums = re.findall(r"\d+", y)
        if len(nums) == 0:
            return 0
        return int(nums[0])