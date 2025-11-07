import os
from typing import Callable, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def apply_notch_filter(eeg_data: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Apply notch filter to remove 50 Hz power line interference.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_channels, n_samples)
        sampling_rate (float): Sampling rate in Hz

    Returns:
        np.ndarray: Filtered EEG data
    """
    b, a = iirnotch(50.0, 30.0, sampling_rate)
    filtered_data = filtfilt(b, a, eeg_data.astype(np.float64), axis=1)
    return filtered_data


def apply_bandpass_filter(eeg_data: np.ndarray, lowpass: float, highpass: float, sampling_rate: float) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_channels, n_samples)
        lowpass (float): Low-pass cutoff frequency in Hz
        highpass (float): High-pass cutoff frequency in Hz
        sampling_rate (float): Sampling rate in Hz

    Returns:
        np.ndarray: Filtered EEG data
    """
    nyquist = sampling_rate / 2
    low = highpass / nyquist
    high = lowpass / nyquist

    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, eeg_data.astype(np.float64), axis=1)

    return filtered_data


def apply_downsample(raw: mne.io.Raw, target_sr: float) -> mne.io.Raw:
    """
    Downsample EEG data using MNE's resample method.

    Args:
        raw (mne.io.Raw): MNE Raw object
        target_sr (float): Target sampling rate

    Returns:
        mne.io.Raw: Downsampled MNE Raw object
    """
    raw_resampled = raw.copy().resample(target_sr)
    return raw_resampled


def apply_common_average_reference(eeg_data: np.ndarray) -> np.ndarray:
    """
    Apply common average reference to EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_channels, n_samples)

    Returns:
        np.ndarray: Rereferenced EEG data
    """
    reference = np.mean(eeg_data, axis=0, keepdims=True)
    rereferenced_data = eeg_data - reference
    return rereferenced_data


class AVEDDataset(BaseDataset):
    r'''
    This dataset contains EEG recordings from 10 subjects listening to one of two competing speech audio streams under audio-video or audio-only conditions. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Zhang et al.
    - Year: 2024
    - Download URL: https://iiphci.ahu.edu.cn/toAuditoryAttention
    - Reference: ZHANG H, ZHANG J. Based on audio-video evoked auditory attention detection electroencephalogram dataset[J]. Journal of Tsinghua University (Science and Technology), 2024, 64(11): 1919-1926.
    - Stimulus: 16 stories selected from the Chinese short story collection "Strange Tales from a Chinese Studio", narrated by a male and a female speaker. Each trial lasts 152 seconds with the target speaker from either 90° left or right.
    - Signals: Electroencephalogram recorded in a double-walled soundproof booth at the Technical University of Denmark (DTU) using a 36-channel system and digitized at a sampling rate of 1000 Hz (downsampled to 128 Hz).
    - Rating: Attended speaker (1 for male, 2 for female), attended direction (1 for left, 2 for right), condition (audio-video or audio-only)

    In order to use this dataset, the download folder :obj:`eeg_raw` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        eeg_raw/
        ├── audio_video
        │   ├── sub1
        │   │   ├── trial1.set
        │   │   ├── trial2.set
        │   │   ├── ...
        │   │   └── trial16.set
        │   ├── sub2
        │   ├── ...
        │   └── sub10
        └── audio_only
            ├── sub1
            │   ├── trial1.set
            │   ├── trial2.set
            │   ├── ...
            │   └── trial16.set
            ├── sub2
            ├── ...
            └── sub10

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AVEDDataset
        from torcheeg import transforms

        dataset = AVEDDataset(root_path='./eeg_raw',
                              chunk_size=128,
                              overlap=0,
                              num_channel=32,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('attended_speaker'),
                                  transforms.Lambda(lambda x: int(x) - 1)
                              ]))
        print(dataset[0])

    Args:
        root_path (str): Path to the raw data files in .set format (unzipped eeg_raw folder) (default: :obj:`'./eeg_raw'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`32`)
        lowpass (float): Low-pass filter cutoff frequency in Hz. (default: :obj:`50.0`)
        highpass (float): High-pass filter cutoff frequency in Hz. (default: :obj:`0.1`)
        sampling_rate (float): Target sampling rate in Hz. (default: :obj:`128.0`)
        online_transform (Callable, optional): The transformation of the EEG signals. The input is a :obj:`np.ndarray`, and the output is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the output is used as the second value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        after_subject (Callable, optional): The hook performed on the subject to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './eeg_raw',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 lowpass: float = 50.0,
                 highpass: float = 0.1,
                 sampling_rate: float = 128.0,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'lowpass': lowpass,
            'highpass': highpass,
            'sampling_rate': sampling_rate,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Dict,
                    root_path: str = './eeg_raw',
                    num_channel: int = 32,
                    lowpass: float = 50.0,
                    highpass: float = 0.1,
                    sampling_rate: float = 128.0,
                    **kwargs) -> Dict:
        condition = record['condition']
        subject_id = record['subject_id']

        subject_dir = os.path.join(root_path, condition, f'sub{subject_id}')

        num_trials = 16
        samples = []

        for trial_id in range(1, num_trials + 1):
            trial_file = f'trial{trial_id}.set'
            trial_path = os.path.join(subject_dir, trial_file)

            if not os.path.exists(trial_path):
                raise FileNotFoundError(f'Trial file not found: {trial_path}')

            raw = mne.io.read_raw_eeglab(
                trial_path, preload=True, verbose=False)
            original_sr = raw.info['sfreq']
            eeg_data = raw.get_data()

            eeg_data = apply_notch_filter(eeg_data, original_sr)

            eeg_data = apply_bandpass_filter(
                eeg_data, lowpass, highpass, original_sr)

            info = mne.create_info(ch_names=[f'CH{i}' for i in range(eeg_data.shape[0])],
                                   sfreq=original_sr,
                                   ch_types='eeg')
            raw_filtered = mne.io.RawArray(eeg_data, info, verbose=False)
            raw_resampled = apply_downsample(raw_filtered, sampling_rate)
            eeg_data = raw_resampled.get_data()

            eeg_data = apply_common_average_reference(eeg_data)

            eeg_data = eeg_data[:num_channel, :]

            samples.append(eeg_data)

        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': condition,
            'subject_id': subject_id,
            'sampling_rate': sampling_rate
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 16
        num_channels = 32
        sampling_rate = kwargs.get('sampling_rate', 128.0)
        trial_length = int(152 * sampling_rate)

        samples = [np.random.randn(num_channels, trial_length)
                   for _ in range(num_trials)]
        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': 'audio_video',
            'subject_id': 1,
            'sampling_rate': sampling_rate
        }

    @staticmethod
    def process_record(record: Dict,
                       samples: list,
                       attended_speaker: np.ndarray,
                       attended_direction: np.ndarray,
                       condition: str,
                       subject_id: int,
                       sampling_rate: float,
                       chunk_size: int = 128,
                       overlap: int = 0,
                       num_channel: int = 32,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        write_pointer = 0

        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id]

            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                'subject_id': subject_id,
                'trial_id': trial_id + 1,
                'attended_speaker': int(attended_speaker[trial_id]),
                'attended_direction': int(attended_direction[trial_id]),
                'condition': condition,
                'sampling_rate': sampling_rate
            }

            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1]
            else:
                dynamic_chunk_size = chunk_size

            start_at = 0
            end_at = start_at + dynamic_chunk_size
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:, start_at:end_at]

                t_eeg = clip_sample

                if offline_transform is not None:
                    t = offline_transform(eeg=clip_sample)
                    t_eeg = t['eeg']

                clip_id = f'{condition}_sub{subject_id}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = './eeg_raw', **kwargs):
        assert os.path.exists(root_path), \
            f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        records = []
        conditions = ['audio_video', 'audio_only']

        for condition in conditions:
            condition_path = os.path.join(root_path, condition)
            assert os.path.exists(condition_path), \
                f'Condition path ({condition_path}) does not exist.'

            subject_dirs = sorted([d for d in os.listdir(condition_path)
                                   if os.path.isdir(os.path.join(condition_path, d))
                                   and d.startswith('sub')])

            for subject_dir in subject_dirs:
                subject_id = int(subject_dir.replace('sub', ''))
                records.append({
                    'condition': condition,
                    'subject_id': subject_id
                })

        return records

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'lowpass': self.lowpass,
                'highpass': self.highpass,
                'sampling_rate': self.sampling_rate,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_subject': self.after_subject,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
