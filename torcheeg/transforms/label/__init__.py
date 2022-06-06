from typing import Dict, List, Union


class Select:
    r'''
    Select part of the value from the information dictionary.

    .. code-block:: python

        transform = Select(key='valence')
        transform({'valence': 4.5, 'arousal': 5.5, 'subject': 7})
        >>> 4.5

    :obj:`Select` allows multiple values to be selected and returned as a list. Suitable for multi-classification tasks or multi-task learning.

    .. code-block:: python

        transform = Select(key=['valence', 'arousal'])
        transform({'valence': 4.5, 'arousal': 5.5, 'subject': 7})
        >>> [4.5, 5.5]

    Args:
        key (str or list): The selected key can be a key string or a list of keys.

    .. automethod:: __call__
    '''
    def __init__(self, key: Union[str, List]):
        self.key = key
        self.select_list = isinstance(key, list) or isinstance(key, tuple)

    def __call__(self, label_dict: Dict) -> Union[int, float, List]:
        r'''
        Args:
            label_dict (dict): A dictionary describing the EEG signal samples, usually as the last return value for each sample in :obj:`Dataset`.
            
        Returns:
            str or list: Selected value or selected value list.
        '''
        assert isinstance(
            label_dict, dict
        ), f'The transform Select only accepts label dict as input, but obtain {type(label_dict)} as input.'
        if self.select_list:
            return [label_dict[k] for k in self.key]
        return label_dict[self.key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Binary:
    r'''
    Binarize the label according to a certain threshold. Labels larger than the threshold are set to 1, and labels smaller than the threshold are set to 0.
    
    .. code-block:: python

        transform = Binary(threshold=5.0)
        transform(4.5)
        >>> 0

    :obj:`Binary` allows simultaneous binarization using the same threshold for multiple labels.

    .. code-block:: python

        transform = Binary(threshold=5.0)
        transform([4.5, 5.5])
        >>> [0, 1]

    Args:
        threshold (float): Threshold used during binarization.

    .. automethod:: __call__
    '''
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, label: Union[int, float, List]) -> Union[int, List]:
        r'''
        Args:
            label (int, float, or list): The input label or list of labels.
            
        Returns:
            int, float, or list: The output label or list of labels after binarization.
        '''
        assert isinstance(
            label, (int, float, list)
        ), f'The transform Binary only accepts label list or item (int or float) as input, but obtain {type(label)} as input.'
        if isinstance(label, list):
            return [int(l >= self.threshold) for l in label]
        return int(label >= self.threshold)


class BinariesToCategory:
    r'''
    Convert multiple binary labels into one multiclass label. Multiclass labels represent permutations of binary labels. Commonly used to combine two binary classification tasks into a single quad classification task.
    
    .. code-block:: python

        transform = BinariesToCategory()
        transform([0, 0])
        >>> 0
        transform([0, 1])
        >>> 1
        transform([1, 0])
        >>> 2
        transform([1, 1])
        >>> 3
    
    .. automethod:: __call__
    '''
    def __call__(self, label_list: List) -> int:
        r'''
        Args:
            label_list (list): list of binary labels.
            
        Returns:
            int: The converted multiclass label.
        '''
        assert isinstance(
            label_list, list
        ), f'The transform BinariesToCategory only accepts label list as input, but obtain {type(label_list)} as input.'
        return sum([v * 2**i for i, v in enumerate(reversed(label_list))])
