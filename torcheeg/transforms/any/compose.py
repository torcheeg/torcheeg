from typing import Callable, List

from ..base_transform import BaseTransform


class Compose(BaseTransform):
    r'''
    Compose several transforms together. Consistent with :obj:`torchvision.transforms.Compose`'s behavior.

    .. code-block:: python

        transform = Compose([
            ToTensor(),
            Resize(size=(64, 64)),
            RandomNoise(p=0.1),
            RandomMask(p=0.1)
        ])
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 64, 64)

    :obj:`Compose` supports transformers with different data dependencies. The above example combines multiple torch-based transformers, the following example shows a sequence of numpy-based transformer.

    .. code-block:: python

        transform = Compose([
            BandDifferentialEntropy(),
            MeanStdNormalize(),
            ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ])
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        transforms (list): The list of transforms to compose.

    .. automethod:: __call__
    '''
    def __init__(self, transforms: List[Callable]):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, *args, **kwargs) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        if args:
            raise KeyError("Please pass data as named parameters.")

        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
