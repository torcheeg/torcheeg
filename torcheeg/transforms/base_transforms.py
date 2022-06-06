from typing import Callable, List


class Lambda:
    r'''
    Apply a user-defined lambda as a transform.

    .. code-block:: python

        transform = Lambda(lambda x: x + 1)
        transform(1)
        >>> 2

    Args:
        lambd (Callable): Lambda/function to be used for transform.

    .. automethod:: __call__
    '''
    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, eeg: any) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        return self.lambd(eeg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose:
    r'''
    Composes several transforms together. Consistent with :obj:`torchvision.transforms.Compose`'s behavior.

    .. code-block:: python

        transform = Composes([
            ToTensor(),
            Resize(size=(64, 64)),
            RandomNoise(p=0.1),
            RandomMask(p=0.1)
        ])
        transform(torch.randn(128, 9, 9)).shape
        >>> (128, 64, 64)

    :obj`Composes` supports transformers with different data dependencies. The above example combines multiple torch-based transformers, the following example shows a sequence of numpy-based transformer.

    .. code-block:: python

        transform = Composes([
            BandDifferentialEntropy(),
            MeanStdNormalize(),
            ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])
        transform(np.random.randn(32, 128)).shape
        >>> (128, 9, 9)

    Args:
        transforms (list): The list of transforms to compose.

    .. automethod:: __call__
    '''
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, eeg: any) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        for t in self.transforms:
            eeg = t(eeg)
        return eeg

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string