import numpy as np


def before_trial_normalize(data: np.ndarray, eps: float = 1e-6):
    r'''
    A common hook function used to normalize the signal of the whole trial before dividing it into chunks.

    It is used as follows:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                before_trial=before_trial_normalize,
                num_worker=4)

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                before_trial=partial(before_trial_normalize, eps=1e-5),
                num_worker=4)

    Args:
        data (np.ndarray): The input EEG signals or features of a trial.
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)
        
    Returns:
        np.ndarray: The normalized results of a trial.
    '''
    min_v = data.min(axis=-1, keepdims=True)
    max_v = data.max(axis=-1, keepdims=True)
    return (data - min_v) / (max_v - min_v + eps)


def after_trial_normalize(data: np.ndarray, eps: float = 1e-6):
    r'''
    A common hook function used to normalize the signal of the whole trial after dividing it into chunks and transforming the divided chunks.

    It is used as follows:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                after_trial=after_trial_normalize,
                num_worker=4)

    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                after_trial=partial(after_trial_normalize, eps=1e-5),
                num_worker=4)
    
    Args:
        data (np.ndarray): The input EEG signals or features of a trial.
        eps (float): The term added to the denominator to improve numerical stability (default: :obj:`1e-6`)
        
    Returns:
        np.ndarray: The normalized results of a trial.
    '''
    trial_samples = []
    trial_keys = []
    trial_infos = []
    for sample in data:
        # electrodes, bands
        trial_samples.append(sample['eeg'])
        trial_keys.append(sample['key'])
        trial_infos.append(sample['info'])

    # windows, electrodes, bands
    trial_samples = np.stack(trial_samples, axis=0)

    min_v = trial_samples.min(axis=0, keepdims=True)
    max_v = trial_samples.max(axis=0, keepdims=True)
    trial_samples = (trial_samples - min_v) / (max_v - min_v + eps)

    output_data = []
    for i, sample in enumerate(trial_samples):
        output_data.append({
            'eeg': sample,
            'key': trial_keys[i],
            'info': trial_infos[i]
        })
    return output_data


def after_trial_moving_avg(data: list, window_size: int = 4):
    '''
    A common hook function for smoothing the signal of each chunk in a trial after pre-processing.

    It is used as follows:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                after_trial=after_trial_moving_avg,
                num_worker=4)
    
    If you want to pass in parameters, use partial to generate a new function:

    .. code-block:: python

        from functools import partial
        dataset = DEAPDataset(
                ...
                after_trial=partial(after_trial_moving_avg, eps=1e-5),
                num_worker=4)

    Args:
        data (np.ndarray): A list of dictionaries, one of which corresponds to an EEG signal in trial. Each dictionary consists of two key-value paris, eeg and key. The value of eeg is the representation of the EEG signal and the value of key is its ID in the IO.
        window_size (int): The window size of moving average. (default: :obj:`4`)
        
    Returns:
        list: The smoothing results of a trial. It is a list of dictionaries, one of which corresponds to an EEG signal in trial. Each dictionary consists of two key-value paris, eeg and key. The value of eeg is the representation of the EEG signal and the value of key is its ID in the IO.
    '''
    trial_samples = []
    trial_keys = []
    trial_infos = []
    for sample in data:
        # electrodes, bands
        trial_samples.append(sample['eeg'])
        trial_keys.append(sample['key'])
        trial_infos.append(sample['info'])

    trial_samples = np.stack(trial_samples, axis=0)
    trial_samples_shape = trial_samples.shape
    trial_samples = trial_samples.reshape(trial_samples.shape[0], -1)
    # windows, electrodes * bands
    trial_samples_T = trial_samples.T

    # electrodes * bands, n
    channel_list = []
    for channel in trial_samples_T:
        moving_avg_channel = np.convolve(channel, np.ones(window_size),
                                         'same') / window_size
        channel_list.append(moving_avg_channel)
    trial_samples_T = np.array(channel_list)

    # windows, electrodes * bands
    trial_samples = trial_samples_T.T
    # windows, electrodes, bands
    trial_samples = trial_samples.reshape(*trial_samples_shape)

    output_data = []
    for i, sample in enumerate(trial_samples):
        output_data.append({
            'eeg': sample,
            'key': trial_keys[i],
            'info': trial_infos[i]
        })
    return output_data
