import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.io import loadmat

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class KULProcessedData(BaseData):
    """Read and preprocess records for :class:`KULProcessedDataset`."""

    channels = [
        'FP1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',  'FT7',
        'FC5', 'FC3', 'FC1', 'C1',  'C3',  'C5',  'T7',  'TP7',
        'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',  'P7',  'P9',
        'PO7', 'PO3', 'O1',  'IZ',  'OZ',  'POZ', 'PZ',  'CPZ',
        'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ',  'F2',  'F4',
        'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ',
        'C2',  'C4',  'C6',  'T8',  'TP8', 'CP6', 'CP4', 'CP2',
        'P2',  'P4',  'P6',  'P8',  'P10', 'PO8', 'PO4', 'O2',
    ]

    grid = [
        ['-',   '-',   '-',   'FP1', 'FPZ', 'FP2', '-',   '-',   '-'],
        ['-',   'AF7', 'AF3', '-',   'AFZ', '-',   'AF4', 'AF8', '-'],
        ['F7',  'F5',  'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8'],
        ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
        ['T7',  'C5',  'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8'],
        ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
        ['P7',  'P5',  'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8'],
        ['P9',  'PO7', 'PO3', '-',   'POZ', '-',   'PO4', 'PO8', 'P10'],
        ['-',   '-',   'O1',  '-',   'OZ',  '-',   'O2',  '-',   '-'],
        ['-',   '-',   '-',   '-',   'IZ',  '-',   '-',   '-',   '-'],
    ]

    adjacency = {
        'AF3': ['FP1', 'AF7', 'F5', 'F3', 'F1'],
        'AF4': ['FP2', 'AF8', 'F2', 'F4', 'F6'],
        'AF7': ['AF3', 'F7', 'F5', 'F3'],
        'AF8': ['AF4', 'F4', 'F6', 'F8'],
        'AFZ': ['FP1', 'FPZ', 'FP2', 'F1', 'FZ', 'F2'],
        'C1':  ['FC3', 'FC1', 'FCZ', 'C3', 'CZ', 'CP3', 'CP1', 'CPZ'],
        'C2':  ['FCZ', 'FC2', 'FC4', 'CZ', 'C4', 'CPZ', 'CP2', 'CP4'],
        'C3':  ['FC5', 'FC3', 'FC1', 'C5', 'C1', 'CP5', 'CP3', 'CP1'],
        'C4':  ['FC2', 'FC4', 'FC6', 'C2', 'C6', 'CP2', 'CP4', 'CP6'],
        'C5':  ['FT7', 'FC5', 'FC3', 'T7', 'C3', 'TP7', 'CP5', 'CP3'],
        'C6':  ['FC4', 'FC6', 'FT8', 'C4', 'T8', 'CP4', 'CP6', 'TP8'],
        'CP1': ['C3', 'C1', 'CZ', 'CP3', 'CPZ', 'P3', 'P1', 'PZ'],
        'CP2': ['CZ', 'C2', 'C4', 'CPZ', 'CP4', 'PZ', 'P2', 'P4'],
        'CP3': ['C5', 'C3', 'C1', 'CP5', 'CP1', 'P5', 'P3', 'P1'],
        'CP4': ['C2', 'C4', 'C6', 'CP2', 'CP6', 'P2', 'P4', 'P6'],
        'CP5': ['T7', 'C5', 'C3', 'TP7', 'CP3', 'P7', 'P5', 'P3'],
        'CP6': ['C4', 'C6', 'T8', 'CP4', 'TP8', 'P4', 'P6', 'P8'],
        'CPZ': ['C1', 'CZ', 'C2', 'CP1', 'CP2', 'P1', 'PZ', 'P2'],
        'CZ':  ['FC1', 'FCZ', 'FC2', 'C1', 'C2', 'CP1', 'CPZ', 'CP2'],
        'F1':  ['AF3', 'AFZ', 'F3', 'FZ', 'FC3', 'FC1', 'FCZ'],
        'F2':  ['AFZ', 'AF4', 'FZ', 'F4', 'FCZ', 'FC2', 'FC4'],
        'F3':  ['AF7', 'AF3', 'F5', 'F1', 'FC5', 'FC3', 'FC1'],
        'F4':  ['AF4', 'AF8', 'F2', 'F6', 'FC2', 'FC4', 'FC6'],
        'F5':  ['AF7', 'AF3', 'F7', 'F3', 'FT7', 'FC5', 'FC3'],
        'F6':  ['AF4', 'AF8', 'F4', 'F8', 'FC4', 'FC6', 'FT8'],
        'F7':  ['AF7', 'F5', 'FT7', 'FC5'],
        'F8':  ['AF8', 'F6', 'FC6', 'FT8'],
        'FC1': ['F3', 'F1', 'FZ', 'FC3', 'FCZ', 'C3', 'C1', 'CZ'],
        'FC2': ['FZ', 'F2', 'F4', 'FCZ', 'FC4', 'CZ', 'C2', 'C4'],
        'FC3': ['F5', 'F3', 'F1', 'FC5', 'FC1', 'C5', 'C3', 'C1'],
        'FC4': ['F2', 'F4', 'F6', 'FC2', 'FC6', 'C2', 'C4', 'C6'],
        'FC5': ['F7', 'F5', 'F3', 'FT7', 'FC3', 'T7', 'C5', 'C3'],
        'FC6': ['F4', 'F6', 'F8', 'FC4', 'FT8', 'C4', 'C6', 'T8'],
        'FCZ': ['F1', 'FZ', 'F2', 'FC1', 'FC2', 'C1', 'CZ', 'C2'],
        'FP1': ['FPZ', 'AF3', 'AFZ'],
        'FP2': ['FPZ', 'AFZ', 'AF4'],
        'FPZ': ['FP1', 'FP2', 'AFZ'],
        'FT7': ['F7', 'F5', 'FC5', 'T7', 'C5'],
        'FT8': ['F6', 'F8', 'FC6', 'C6', 'T8'],
        'FZ':  ['AFZ', 'F1', 'F2', 'FC1', 'FCZ', 'FC2'],
        'IZ':  ['OZ'],
        'O1':  ['PO7', 'PO3'],
        'O2':  ['PO4', 'PO8'],
        'OZ':  ['POZ', 'IZ'],
        'P1':  ['CP3', 'CP1', 'CPZ', 'P3', 'PZ', 'PO3', 'POZ'],
        'P10': ['P6', 'P8', 'PO8'],
        'P2':  ['CPZ', 'CP2', 'CP4', 'PZ', 'P4', 'POZ', 'PO4'],
        'P3':  ['CP5', 'CP3', 'CP1', 'P5', 'P1', 'PO7', 'PO3'],
        'P4':  ['CP2', 'CP4', 'CP6', 'P2', 'P6', 'PO4', 'PO8'],
        'P5':  ['TP7', 'CP5', 'CP3', 'P7', 'P3', 'P9', 'PO7', 'PO3'],
        'P6':  ['CP4', 'CP6', 'TP8', 'P4', 'P8', 'PO4', 'PO8', 'P10'],
        'P7':  ['TP7', 'CP5', 'P5', 'P9', 'PO7'],
        'P8':  ['CP6', 'TP8', 'P6', 'PO8', 'P10'],
        'P9':  ['P7', 'P5', 'PO7'],
        'PO3': ['P5', 'P3', 'P1', 'PO7', 'O1'],
        'PO4': ['P2', 'P4', 'P6', 'PO8', 'O2'],
        'PO7': ['P7', 'P5', 'P3', 'P9', 'PO3', 'O1'],
        'PO8': ['P4', 'P6', 'P8', 'PO4', 'P10', 'O2'],
        'POZ': ['P1', 'PZ', 'P2', 'OZ'],
        'PZ':  ['CP1', 'CPZ', 'CP2', 'P1', 'P2', 'POZ'],
        'T7':  ['FT7', 'FC5', 'C5', 'TP7', 'CP5'],
        'T8':  ['FC6', 'FT8', 'C6', 'CP6', 'TP8'],
        'TP7': ['T7', 'C5', 'CP5', 'P7', 'P5'],
        'TP8': ['C6', 'T8', 'CP6', 'P6', 'P8'],
    }

    def __init__(self,
                 root_path: str = './preprocessed_data',
                 chunk_size: int = 64,
                 overlap: int = 0,
                 num_channel: int = 64,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel
        )

    def read_record(self,
                    record: str,
                    root_path: str = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        mat_path = os.path.join(root_path, record)
        matstruct_contents = loadmat(mat_path)

        n_trials = 8

        eeg_data = []
        attended_ear_data = []
        condition_data = []

        for trial_id in range(n_trials):
            trial = matstruct_contents['preproc_trials'][0, trial_id]

            eeg = trial[0, 0][0][0, 0][1]
            attended_ear = trial[0, 0][3][0]
            condition = trial[0, 0][5][0]

            eeg_data.append(eeg)
            attended_ear_data.append(attended_ear)
            condition_data.append(condition)

        return {
            'samples': eeg_data,
            'attended_ear': attended_ear_data,
            'condition': condition_data
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 20
        num_channels = 64
        trial_length = 12448

        samples = [np.random.randn(trial_length, num_channels)
                   for _ in range(num_trials)]
        attended_ear = [np.random.choice(['L', 'R'])
                        for _ in range(num_trials)]
        condition = [np.random.choice(['dry', 'hrtf'])
                     for _ in range(num_trials)]

        return {
            'samples': samples,
            'attended_ear': attended_ear,
            'condition': condition
        }

    def process_record(self, record: str, samples: list, attended_ear: list, condition: list, chunk_size: int = None, overlap: int = None, num_channel: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        subject_id = record.split('.')[0]
        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id][:, :num_channel].T
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id,
                               'attended_ear': attended_ear[trial_id], 'condition': condition[trial_id]}
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

        records = [f for f in os.listdir(root_path) if f.endswith(
            '.mat') and f.startswith('S')]
        return sorted(records)


class KULProcessedDataset(CachedDataset):

    r"""
    This dataset contains EEG recordings from 16 subjects listening to one of two competing speech audio streams, after matlab preprocessing. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Biesmans et al.
    - Year: 2016
    - Download URL: https://zenodo.org/records/4004271
    - Reference: Biesmans W, Das N, Francart T, et al. Auditory-inspired speech envelope extraction methods for improved EEG-based auditory attention detection in a cocktail party scenario[J]. IEEE transactions on neural systems and rehabilitation engineering, 2016, 25(5): 402-412.
    - Stimulus: Four Dutch short stories, lasting approximately six minutes each. Either dry speech was offered, i.e., each speaker was presented to a different ear, or speech signals were processed by (dead room) head-related transfer functions, simulating a more realistic listening scenario in which the speakers are spatially located 90 degrees to the left and the right of the subject.
    - Signals: The BioSemi ActiveTwo system was used to record 64-channel EEG signals at 8196 Hz sample rate (downsampled to 64 Hz). Each subject completed 8 trials, each lasting 6 minutes.
    - Labels: Attended direction (attended_ear='L' for left, attended_ear='R' for right), stimulus condition (condition='dry' or 'hrtf').

    In order to use this dataset, the download file :obj:`4004271.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        4004271/
        ├── preprocess_data.m
        ├── S1.mat
        ├── S2.mat
        ├── S3.mat
        ├── ...
        └── S16.mat

    After running preprocess_data.m, preprocessed_data folder should be generated and contain the following files:

    .. code-block:: text

        preprocessed_data/
        ├── S1.mat
        ├── S2.mat
        ├── S3.mat
        ├── ...
        └── S16.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import KULProcessedDataset
        from torcheeg import transforms


        def preprocess_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.To2d(),
            ])
            return {
                **sample,
                'eeg': eeg_transform(sample['eeg']),
            }


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('attended_ear'),
                transforms.Lambda(lambd=lambda x: 1 if x == 'L' else 0),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = KULProcessedDataset(
            root_path='./preprocessed_data',
            offline_transform=preprocess_fn,
            online_transform=transform_fn,
        )
        print(dataset[0])

    Args:
        root_path (str): Path to the preprocessed data files in MATLAB format. (default: :obj:`'./preprocessed_data'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of an entire trial is used as a single chunk. (default: :obj:`64`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG data. (default: :obj:`0`)
        num_channel (int): Number of channels to use. The dataset contains 64 channels. (default: :obj:`64`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to the generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size the database may grow to; used to size the memory mapping. If the database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signals. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system-based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory is used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars. (default: :obj:`True`)
    """

    data_class = KULProcessedData

    channels = KULProcessedData.channels

    grid = KULProcessedData.grid

    adjacency = KULProcessedData.adjacency

    def __init__(self, root_path: str = './preprocessed_data', chunk_size: int = 64, overlap: int = 0, num_channel: int = 64, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel,
                  'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
