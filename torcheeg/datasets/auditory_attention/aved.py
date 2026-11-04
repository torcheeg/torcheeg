import os
from typing import Callable, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


def _apply_notch_filter(eeg_data: np.ndarray, sampling_rate: float) -> np.ndarray:
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


def _apply_bandpass_filter(eeg_data: np.ndarray, lowpass: float, highpass: float, sampling_rate: float) -> np.ndarray:
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


def _apply_downsample(raw: mne.io.Raw, target_sr: float) -> mne.io.Raw:
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


def _apply_common_average_reference(eeg_data: np.ndarray) -> np.ndarray:
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


class AVEDData(BaseData):
    """Read and preprocess records for :class:`AVEDDataset`."""

    channels = [
        'FP1', 'FP2', 'F7',  'F3',  'FZ',  'F4',  'F8',  'FT7',
        'FC3', 'FCZ', 'FC4', 'FT8', 'T3',  'C3',  'CZ',  'C4',
        'T4',  'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'A1',  'T5',
        'P3',  'PZ',  'P4',  'T6',  'A2',  'O1',  'OZ',  'O2',
    ]

    grid = [
        ['-',   '-', '-',   'FP1', '-',   'FP2', '-',   '-', '-'],
        ['F7',  '-', 'F3',  '-',   'FZ',  '-',   'F4',  '-', 'F8'],
        ['FT7', '-', 'FC3', '-',   'FCZ', '-',   'FC4', '-', 'FT8'],
        ['T3',  '-', 'C3',  '-',   'CZ',  '-',   'C4',  '-', 'T4'],
        ['TP7', '-', 'CP3', '-',   'CPZ', '-',   'CP4', '-', 'TP8'],
        ['T5',  '-', 'P3',  '-',   'PZ',  '-',   'P4',  '-', 'T6'],
        ['-',   '-', '-',   'O1',  'OZ',  'O2',  '-',   '-', '-'],
        ['A1',  '-', '-',   '-',   '-',   '-',   '-',   '-', 'A2'],
    ]

    adjacency = {
        'A1':  [],
        'A2':  [],
        'C3':  ['FC3', 'CP3'],
        'C4':  ['FC4', 'CP4'],
        'CP3': ['C3', 'P3'],
        'CP4': ['C4', 'P4'],
        'CPZ': ['CZ', 'PZ'],
        'CZ':  ['FCZ', 'CPZ'],
        'F3':  ['FP1', 'FC3'],
        'F4':  ['FP2', 'FC4'],
        'F7':  ['FT7'],
        'F8':  ['FT8'],
        'FC3': ['F3', 'C3'],
        'FC4': ['F4', 'C4'],
        'FCZ': ['FZ', 'CZ'],
        'FP1': ['F3', 'FZ'],
        'FP2': ['FZ', 'F4'],
        'FT7': ['F7', 'T3'],
        'FT8': ['F8', 'T4'],
        'FZ':  ['FP1', 'FP2', 'FCZ'],
        'O1':  ['P3', 'PZ', 'OZ'],
        'O2':  ['PZ', 'P4', 'OZ'],
        'OZ':  ['PZ', 'O1', 'O2'],
        'P3':  ['CP3', 'O1'],
        'P4':  ['CP4', 'O2'],
        'PZ':  ['CPZ', 'O1', 'OZ', 'O2'],
        'T3':  ['FT7', 'TP7'],
        'T4':  ['FT8', 'TP8'],
        'T5':  ['TP7'],
        'T6':  ['TP8'],
        'TP7': ['T3', 'T5'],
        'TP8': ['T4', 'T6'],
    }

    raw_channels = ['HEOL', 'HEOR', 'FP1', 'FP2', 'VEOU', 'VEOL', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4',
                    'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4', 'T6', 'A2', 'O1', 'OZ', 'O2']

    def __init__(self,
                 root_path: str = './eeg_raw',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 lowpass: float = 50.0,
                 highpass: float = 0.1,
                 sampling_rate: float = 128.0,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel,
            lowpass=lowpass,
            highpass=highpass,
            sampling_rate=sampling_rate
        )

    def read_record(self,
                    record: Dict,
                    root_path: str = None,
                    num_channel: int = None,
                    lowpass: float = None,
                    highpass: float = None,
                    sampling_rate: float = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        if num_channel is None:
            num_channel = self.num_channel
        if lowpass is None:
            lowpass = self.lowpass
        if highpass is None:
            highpass = self.highpass
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
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
            raw.rename_channels({name: name.upper() for name in raw.ch_names})
            raw.pick(AVEDData.channels[:num_channel])
            original_sr = raw.info['sfreq']
            eeg_data = raw.get_data()

            eeg_data = _apply_notch_filter(eeg_data, original_sr)

            eeg_data = _apply_bandpass_filter(
                eeg_data, lowpass, highpass, original_sr)

            info = mne.create_info(ch_names=raw.ch_names,
                                   sfreq=original_sr,
                                   ch_types='eeg')
            raw_filtered = mne.io.RawArray(eeg_data, info, verbose=False)
            raw_resampled = _apply_downsample(raw_filtered, sampling_rate)
            eeg_data = raw_resampled.get_data()

            eeg_data = _apply_common_average_reference(eeg_data)

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

    def process_record(self, record: Dict, samples: list, attended_speaker: np.ndarray, attended_direction: np.ndarray, condition: str, subject_id: int, sampling_rate: float = None, chunk_size: int = None, overlap: int = None, num_channel: int = None, **kwargs):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id]
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id + 1, 'attended_speaker': int(
                attended_speaker[trial_id]), 'attended_direction': int(attended_direction[trial_id]), 'condition': condition, 'sampling_rate': sampling_rate}
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
                record_info = {'start_at': start_at, 'end_at': end_at}
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'info': record_info}
                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )

        records = []
        conditions = ['audio_video', 'audio_only']

        for condition in conditions:
            condition_path = os.path.join(root_path, condition)
            if not (os.path.exists(condition_path)):
                raise FileNotFoundError(
                    f'Condition path ({condition_path}) does not exist.')

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


class AVEDDataset(CachedDataset):

    r"""
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


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('attended_speaker'),
                transforms.Lambda(lambda x: int(x) - 1),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = AVEDDataset(
            root_path='./eeg_raw',
            chunk_size=128,
            overlap=0,
            num_channel=32,
            online_transform=transform_fn,
        )
        print(dataset[0])

    Args:
        root_path (str): Path to the raw data files in .set format (unzipped eeg_raw folder) (default: :obj:`'./eeg_raw'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`32`)
        lowpass (float): Low-pass filter cutoff frequency in Hz. (default: :obj:`50.0`)
        highpass (float): High-pass filter cutoff frequency in Hz. (default: :obj:`0.1`)
        sampling_rate (float): Target sampling rate in Hz. (default: :obj:`128.0`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    raw_channels = AVEDData.raw_channels

    data_class = AVEDData

    channels = AVEDData.channels

    grid = AVEDData.grid

    adjacency = AVEDData.adjacency

    def __init__(self, root_path: str = './eeg_raw', chunk_size: int = 128, overlap: int = 0, num_channel: int = 32, lowpass: float = 50.0, highpass: float = 0.1, sampling_rate: float = 128.0, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 10485760, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel, 'lowpass': lowpass, 'highpass': highpass,
                  'sampling_rate': sampling_rate, 'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'lowpass': self.lowpass, 'highpass': self.highpass, 'sampling_rate': self.sampling_rate, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
