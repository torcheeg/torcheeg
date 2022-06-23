from typing import Callable, Dict, List

from ..base_transform import BaseTransform


class Lambda(BaseTransform):
    r'''
    Apply a user-defined lambda as a transform.

    .. code-block:: python

        transform = Lambda(targets=['y'], lambda x: x + 1)
        transform(y=1)['y']
        >>> 2

    Args:
        targets (list): What data to transform via the Lambda. (default: :obj:`['eeg', 'baseline', 'y']`)
        lambd (Callable): Lambda/function to be used for transform.

    .. automethod:: __call__
    '''
    def __init__(self,
                 lambd: Callable,
                 targets: List[str] = ['eeg', 'baseline', 'y']):
        super(Lambda, self).__init__()
        self._targets = targets
        self.lambd = lambd

    @property
    def targets(self) -> Dict[str, Callable]:
        return {target: self.apply for target in self._targets}

    def apply(self, *args, **kwargs) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The transformed output.
        '''
        return self.lambd(args[0])

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            x (any): The input.
        Returns:
            any: The transformed output.
        '''
        return super().__call__(*args, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'lambd': self.lambd,
            'targets': [...]
        })