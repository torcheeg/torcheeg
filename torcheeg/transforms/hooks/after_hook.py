import numpy as np
import torch

from typing import List, Union


def after_hook_normalize(
        data: List[Union[np.ndarray, torch.Tensor]],
        eps: float = 1e-6) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_normalize
        
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_normalize,
                              num_worker=4,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_normalize

        DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_normalize, eps=1e-5),
                              num_worker=4,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)
        
    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        data = np.stack(data, axis=0)

        min_v = data.min(axis=0, keepdims=True)
        max_v = data.max(axis=0, keepdims=True)
        data = (data - min_v) / (max_v - min_v + eps)

        return [sample for sample in data]
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=0)

        min_v, _ = data.min(axis=0, keepdims=True)
        max_v, _ = data.max(axis=0, keepdims=True)
        data = (data - min_v) / (max_v - min_v + eps)

        return [sample for sample in data]
    else:
        raise ValueError(
            'The after_hook_normalize only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))


def after_hook_running_norm(
        data: List[Union[np.ndarray, torch.Tensor]],
        decay_rate: float = 0.9,
        eps: float = 1e-6) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_running_norm
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_running_norm,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_running_norm
        
        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_running_norm, decay_rate=0.9, eps=1e-6),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))
    
    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        decay_rate (float): The decay rate used in the running normalization (default: :obj:`0.9`)
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)

    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        data = np.stack(data, axis=0)

        running_mean = np.zeros_like(data[0])
        running_var = np.zeros_like(data[0])

        for i, current_sample in enumerate(data):
            running_mean = decay_rate * running_mean + (
                1 - decay_rate) * current_sample
            running_var = decay_rate * running_var + (
                1 - decay_rate) * np.square(current_sample - running_mean)
            data[i] = (data[i] - running_mean) / np.sqrt(running_var + eps)

        return [sample for sample in data]
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=0)

        running_mean = torch.zeros_like(data[0])
        running_var = torch.zeros_like(data[0])

        for i, current_sample in enumerate(data):
            running_mean = decay_rate * running_mean + (
                1 - decay_rate) * current_sample
            running_var = decay_rate * running_var + (
                1 - decay_rate) * torch.square(current_sample - running_mean)
            data[i] = (data[i] - running_mean) / torch.sqrt(running_var + eps)

        return [sample for sample in data]
    else:
        raise ValueError(
            'The after_hook_running_norm only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))


def after_hook_linear_dynamical_system(
        data: List[Union[np.ndarray, torch.Tensor]],
        V0: float = 0.01,
        A: float = 1,
        T: float = 0.0001,
        C: float = 1,
        sigma: float = 1) -> List[Union[np.ndarray, torch.Tensor]]:
    r'''
    A common hook function used to normalize the signal of the whole trial/session/subject after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_linear_dynamical_system

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=after_hook_linear_dynamical_system,
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        from torcheeg.datasets import DEAPDataset
        from torcheeg.transforms import after_hook_linear_dynamical_system

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              after_trial=partial(after_hook_linear_dynamical_system, V0=0.01, A=1, T=0.0001, C=1, sigma=1),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))
    
    Args:
        data (list): A list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
        V0 (float): The initial variance of the linear dynamical system (default: :obj:`0.01`)
        A (float): The coefficient of the linear dynamical system (default: :obj:`1`)
        T (float): The term added to the diagonal of the covariance matrix (default: :obj:`0.0001`)
        C (float): The coefficient of the linear dynamical system (default: :obj:`1`)
        sigma (float): The variance of the linear dynamical system (default: :obj:`1`)
        
    Returns:
        list: The normalized results of a trial. It is a list of :obj:`np.ndarray` or :obj:`torch.Tensor`, one of which corresponds to an EEG signal in trial.
    '''
    if isinstance(data[0], np.ndarray):
        # save the data[0].shape and flatten them
        shape = data[0].shape
        data = np.stack([sample.flatten() for sample in data], axis=0)

        ave = np.mean(data, axis=0)
        u0 = ave
        X = data.transpose((1, 0))

        [m, n] = X.shape
        P = np.zeros((m, n))
        u = np.zeros((m, n))
        V = np.zeros((m, n))
        K = np.zeros((m, n))

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m, ))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (np.ones((m, )) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:,
              i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (np.ones((m, )) - K[:, i] * C) * P[:, i - 1]

        X = u

        return [sample.reshape(shape) for sample in X.transpose((1, 0))]

    elif isinstance(data[0], torch.Tensor):
        shape = data[0].shape
        data = torch.stack([sample.flatten() for sample in data], dim=0)

        ave = torch.mean(data, dim=0)
        u0 = ave
        X = data.transpose(1, 0)

        [m, n] = X.shape
        P = torch.zeros((m, n))
        u = torch.zeros((m, n))
        V = torch.zeros((m, n))
        K = torch.zeros((m, n))

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * torch.ones((m, ))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (torch.ones((m, )) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:,
              i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (torch.ones((m, )) - K[:, i] * C) * P[:, i - 1]

        X = u

        return [sample.reshape(shape) for sample in X.transpose(1, 0)]

    else:
        raise ValueError(
            'The after_hook_linear_dynamical_system only supports np.ndarray and torch.Tensor. Please make sure the outputs of offline_transform ({}) are np.ndarray or torch.Tensor.'
            .format(type(data[0])))