import pdb
from typing import Any

from ..base_transform import BaseTransform


class PDB(BaseTransform):
    """Pause in the debugger before returning the input unchanged."""

    def __call__(self, x: Any) -> Any:
        pdb.set_trace()
        return x
