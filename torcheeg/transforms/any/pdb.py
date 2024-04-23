import pdb
from typing import Dict, List, Callable

from ..base_transform import BaseTransform


class PDB(BaseTransform):
    r'''
    For debugging, insert breakpoints in transforms. The transformation itself does not change the input data.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(64, 64)),
            transforms.PDB(),
            transforms.RandomNoise(p=0.1),
            transforms.RandomMask(p=0.1)
        ])
        t(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 64, 64)
    
    Args:
        targets (list): What data to transform via the PDB. (default: :obj:`['eeg', 'baseline', 'y']`)

    .. automethod:: __call__
    '''
    def __init__(self, targets: List[str] = ['eeg', 'baseline', 'y']):
        super(PDB, self).__init__()
        self._targets = targets

    def apply(self, *args, **kwargs) -> any:
        r'''
        Args:
            x (any): The input.

        Returns:
            any: The input.
        '''
        return args[0]

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        r'''
        Args:
            x (any): The input.
        Returns:
            any: The input.
        '''
        pdb.set_trace()
        return super().__call__(*args, **kwargs)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {target: self.apply for target in self._targets}

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'targets': [...]})
