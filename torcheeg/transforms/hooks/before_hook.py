import numpy as np


def before_hook_normalize(data: np.ndarray,
                          eps: float = 1e-6,
                          axis=0) -> np.ndarray:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject before dividing it into chunks.

    It is used as follows:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import SEEDFeatureDataset
        from torcheeg.transforms import before_hook_normalize

        dataset = SEEDFeatureDataset(root_path='./ExtractedFeatures',
                                     feature=['de_movingAve'],
                                     offline_transform=transforms.ToGrid       (SEED_CHANNEL_LOCATION_DICT),
                                     online_transform=transforms.ToTensor(),
                                     before_trial=before_hook_normalize,
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion'),
                                         transforms.Lambda(lambda x: x + 1)
                                     ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import SEEDFeatureDataset
        from torcheeg.transforms import before_hook_normalize

        dataset = SEEDFeatureDataset(root_path='./ExtractedFeatures',
                                     feature=['de_movingAve'],
                                     offline_transform=transforms.ToGrid       (SEED_CHANNEL_LOCATION_DICT),
                                     online_transform=transforms.ToTensor(),
                                     before_trial=partial(before_hook_normalize, eps=1e-5),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion'),
                                         transforms.Lambda(lambda x: x + 1)
                                     ]))

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