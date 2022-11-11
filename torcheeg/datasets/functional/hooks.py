from statsmodels.tsa.vector_ar.var_model import VAR

import numpy as np


def before_trial_normalize(data, eps=1e-6):
    '''
    一个常用的钩子函数，用于在将实验划分为 chunk 之前，对整个实验的信号进行归一化。
    使用方法为：

    ```
    from functools import partial
    dataset = DEAPDataset(
            ...
            before_trial=partial(before_trial_normalize, eps=1e-6),
            num_worker=4)
    ```
    
    参数：
    numpy 数组：一个 trial 的信号，允许多维矩阵作为输入，其中最后一维表示时序数据点。
    返回：
    numpy 数组：归一化后一个 trial 的信号
    '''
    min_v = data.min(axis=-1, keepdims=True)
    max_v = data.max(axis=-1, keepdims=True)
    return (data - min_v) / (max_v - min_v + eps)


def after_trial_normalize(data, eps=1e-6):
    trial_samples = []
    trial_keys = []
    for sample in data:
        # electrodes, bands
        trial_samples.append(sample['eeg'])
        trial_keys.append(sample['key'])

    # windows, electrodes, bands
    trial_samples = np.stack(trial_samples, axis=0)

    min_v = trail_samples.min(axis=0, keepdims=True)
    max_v = trail_samples.max(axis=0, keepdims=True)
    trail_samples = (trail_samples - min_v) / (max_v - min_v + eps)

    output_data = []
    for i, sample in enumerate(trial_samples):
        output_data.append({'eeg': sample, 'key': trial_keys[i]})
    return output_data


def after_trial_moving_avg(data, window_size=4):
    '''
    一个常用的钩子函数，用于在将实验划分为 chunk，并对 chunk 信号进行预处理后的特征进行平滑化。
    使用方法为：

    ```
    from functools import partial
    dataset = DEAPDataset(
            ...
            after_trial=partial(after_trial_moving_avg, eps=1e-6),
            num_worker=4)
    ```
    
    参数：
    字典数组：一个字典，其中 ...。
    返回：
    字典数组：一个字典，其中 ...。
    '''
    trial_samples = []
    trial_keys = []
    for sample in data:
        # electrodes, bands
        trial_samples.append(sample['eeg'])
        trial_keys.append(sample['key'])

    trial_samples = np.stack(trial_samples, axis=0)
    trail_samples_shape = trial_samples.shape
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
    trial_samples = trial_samples.reshape(*trail_samples_shape)

    output_data = []
    for i, sample in enumerate(trial_samples):
        output_data.append({'eeg': sample, 'key': trial_keys[i]})
    return output_data
