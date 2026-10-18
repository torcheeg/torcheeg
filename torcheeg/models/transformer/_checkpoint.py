from argparse import Namespace
from contextlib import nullcontext

import torch


def _load_checkpoint(checkpoint_path, allow_training_metadata=False):
    if not isinstance(allow_training_metadata, bool):
        raise ValueError('allow_training_metadata must be a boolean.')
    context = nullcontext()
    if allow_training_metadata:
        if not hasattr(torch.serialization, 'safe_globals'):
            raise RuntimeError('This runtime lacks safe_globals; first convert the verified checkpoint to tensor-only state.')
        import numpy as np
        scalar = (np._core if hasattr(np, '_core') else np.core).multiarray.scalar
        allowed = [Namespace, scalar, (scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype('float64'))]
        existing = torch.serialization.get_safe_globals()
        context = torch.serialization.safe_globals([value for value in allowed if value not in existing])
    with context:
        return torch.load(checkpoint_path, weights_only=True, map_location='cpu')
