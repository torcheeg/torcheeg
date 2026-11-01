import numpy as np


def before_hook_normalize(data: np.ndarray,
                          eps: float = 1e-6,
                          axis=0) -> np.ndarray:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject before dividing it into chunks.

    .. code-block:: python

        import numpy as np
        from torcheeg.transforms import before_hook_normalize

        result = before_hook_normalize(np.arange(8).reshape(2, 4))

    Args:
        data (np.ndarray): The input EEG signals or features of a trial.
        axis (int): The axis along which to normalize the data (default: :obj:`0`)
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)

    Returns:
        np.ndarray: The normalized results of a trial.
    '''
    min_v = data.min(axis=axis, keepdims=True)
    max_v = data.max(axis=axis, keepdims=True)
    return (data - min_v) / (max_v - min_v + eps)
