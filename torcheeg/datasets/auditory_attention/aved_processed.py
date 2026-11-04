import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class AVEDProcessedData(BaseData):
    """Read and preprocess records for :class:`AVEDProcessedDataset`."""

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

    def __init__(self,
                 root_path: str = './eeg_preproc',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel
        )

    def read_record(self,
                    record: Dict,
                    root_path: str = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        condition = record['condition']
        subject_id = record['subject_id']

        filename = os.path.join(root_path, condition, f'sub{subject_id}.csv')
        data_df = pd.read_csv(filename, header=None)
        samples = data_df.values

        trial_length = 152 * 128
        num_trials = 16

        samples = samples.reshape(num_trials, trial_length, -1)
        samples = samples.transpose(0, 2, 1)

        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': condition,
            'subject_id': subject_id
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 16
        num_channels = 32
        trial_length = 152 * 128

        samples = np.random.randn(num_trials, num_channels, trial_length)
        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': 'audio-video',
            'subject_id': 1
        }

    def process_record(self, record: Dict, samples: np.ndarray, attended_speaker: np.ndarray, attended_direction: np.ndarray, condition: str, subject_id: int, chunk_size: int = None, overlap: int = None, num_channel: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, :num_channel, :]
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id + 1, 'attended_speaker': int(
                attended_speaker[trial_id]), 'attended_direction': int(attended_direction[trial_id]), 'condition': condition}
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
        conditions = ['audio-video', 'audio-only']

        for condition in conditions:
            condition_path = os.path.join(root_path, condition)
            if not (os.path.exists(condition_path)):
                raise FileNotFoundError(
                    f'Condition path ({condition_path}) does not exist.')

            files = sorted([f for f in os.listdir(
                condition_path) if f.endswith('.csv')])

            for filename in files:
                subject_id = int(filename.replace(
                    'sub', '').replace('.csv', ''))
                records.append({
                    'condition': condition,
                    'subject_id': subject_id
                })

        return records


class AVEDProcessedDataset(CachedDataset):

    r"""
    This dataset contains EEG recordings from 10 subjects listening to one of two competing speech audio streams under audio-video or audio-only conditions. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Zhang et al.
    - Year: 2024
    - Download URL: https://iiphci.ahu.edu.cn/toAuditoryAttention
    - Reference: ZHANG H, ZHANG J. Based on audio-video evoked auditory attention detection electroencephalogram dataset[J]. Journal of Tsinghua University (Science and Technology), 2024, 64(11): 1919-1926.
    - Stimulus: 16 stories selected from the Chinese short story collection "Strange Tales from a Chinese Studio", narrated by a male and a female speaker. Each trial lasts 152 seconds with the target speaker from either 90° left or right.
    - Signals: Electroencephalogram recorded in a double-walled soundproof booth at the Technical University of Denmark (DTU) using a 32-channel system and digitized at a sampling rate of 1000 Hz (downsampled to 128 Hz).
    - Rating: Attended speaker (1 for male, 2 for female), attended direction (1 for left, 2 for right), condition (audio-video or audio-only)

    In order to use this dataset, the download folder :obj:`eeg_preproc.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        eeg_preproc/
        ├── audio-video
        │   ├── sub1.csv
        │   ├── sub2.csv
        │   ├── sub3.csv
        │   ├── ...
        │   └── sub10.csv
        └── audio-only
            ├── sub1.csv
            ├── sub2.csv
            ├── sub3.csv
            ├── ...
            └── sub10.csv

    In the dataset, 50 Hz power line interference is removed, and the signals are band-pass filtered (0.1-50 Hz) and downsampled to 128 Hz. Subsequently, ocular and muscle artifacts are eliminated using independent component analysis (ICA). Finally, all EEG channels were re-referenced.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AVEDProcessedDataset
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


        dataset = AVEDProcessedDataset(
            root_path='./eeg_preproc',
            chunk_size=128,
            overlap=0,
            num_channel=32,
            online_transform=transform_fn,
        )
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in CSV format (unzipped eeg_preproc folder) (default: :obj:`'./eeg_preproc'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`32`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = AVEDProcessedData

    channels = AVEDProcessedData.channels

    grid = AVEDProcessedData.grid

    adjacency = AVEDProcessedData.adjacency

    def __init__(self, root_path: str = './eeg_preproc', chunk_size: int = 128, overlap: int = 0, num_channel: int = 32, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel,
                  'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
