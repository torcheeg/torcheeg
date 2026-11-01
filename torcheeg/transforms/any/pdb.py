import pdb

from ..base_transform import BaseTransform


class PDB(BaseTransform):
    """Pause in the debugger before returning the input unchanged."""

    def __call__(self, x):
        pdb.set_trace()
        return x
