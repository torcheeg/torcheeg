import os
import pickle as pkl
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset

_VALENCE_DICT = {
    1: -1,  # negative
    2: -1,
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: 0,  # neutral
    14: 0,
    15: 0,
    16: 0,
    17: 1,  # positive
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1
}

_EMOTION_DICT = {
    1: 0,  # anger
    2: 0,
    3: 0,
    4: 1,  # disgust
    5: 1,
    6: 1,
    7: 2,  # fear
    8: 2,
    9: 2,
    10: 3,  # sadness
    11: 3,
    12: 3,
    13: 4,  # neutral
    14: 4,
    15: 4,
    16: 4,
    17: 5,  # amusement
    18: 5,
    19: 5,
    20: 6,  # inspiration
    21: 6,
    22: 6,
    23: 7,  # joy
    24: 7,
    25: 7,
    26: 8,  # tenderness
    27: 8,
    28: 8
}


class FACEDData(BaseData):
    """Read and preprocess records for :class:`FACEDDataset`."""

    channels = [
        'FP1', 'FP2', 'FZ',  'F3',  'F4',  'F7',  'F8',  'FC1',
        'FC2', 'FC5', 'FC6', 'CZ',  'C3',  'C4',  'T7',  'T8',
        'CP1', 'CP2', 'CP5', 'CP6', 'PZ',  'P3',  'P4',  'P7',
        'P8',  'PO3', 'PO4', 'OZ',  'O1',  'O2',
    ]

    grid = [
        ['-',  '-',   '-',  'FP1', '-',  'FP2', '-',  '-',   '-'],
        ['F7', '-',   'F3', '-',   'FZ', '-',   'F4', '-',   'F8'],
        ['-',  'FC5', '-',  'FC1', '-',  'FC2', '-',  'FC6', '-'],
        ['T7', '-',   'C3', '-',   'CZ', '-',   'C4', '-',   'T8'],
        ['-',  'CP5', '-',  'CP1', '-',  'CP2', '-',  'CP6', '-'],
        ['P7', '-',   'P3', '-',   'PZ', '-',   'P4', '-',   'P8'],
        ['-',  '-',   '-',  'PO3', '-',  'PO4', '-',  '-',   '-'],
        ['-',  '-',   '-',  'O1',  'OZ', 'O2',  '-',  '-',   '-'],
    ]

    adjacency = {
        'FP1': ['F3', 'FZ'],
        'FP2': ['FZ', 'F4'],
        'F7':  ['FC5'],
        'F3':  ['FC1', 'FC5'],
        'FZ':  ['AF4', 'FC2', 'FC1'],
        'F4':  ['AF4', 'FC6', 'FC2'],
        'F8':  ['FC6'],
        'FC5': ['F7', 'F3', 'C3', 'T7'],
        'FC1': ['F3', 'FZ', 'CZ', 'C3'],
        'FC2': ['FZ', 'F4', 'C4', 'CZ'],
        'FC6': ['F4', 'F8', 'T8', 'C4'],
        'T7':  ['FC5', 'CP5'],
        'C3':  ['FC5', 'FC1', 'CP1', 'CP5'],
        'CZ':  ['FC1', 'FC2', 'CP2', 'CP1'],
        'C4':  ['FC2', 'FC6', 'CP6', 'CP2'],
        'T8':  ['FC6', 'CP6'],
        'CP5': ['T7', 'C3', 'P3', 'P7'],
        'CP1': ['C3', 'CZ', 'PZ', 'P3'],
        'CP2': ['CZ', 'C4', 'P4', 'PZ'],
        'CP6': ['C4', 'T8', 'P8', 'P4'],
        'P7':  ['CP5'],
        'P3':  ['CP5', 'CP1', 'PO3'],
        'PZ':  ['CP1', 'CP2', 'PO4', 'PO3'],
        'P4':  ['CP2', 'CP6', 'PO4'],
        'P8':  ['CP6'],
        'PO3': ['P3', 'PZ', 'OZ', 'O1'],
        'PO4': ['PZ', 'P4', 'O2', 'OZ'],
        'O1':  ['PO3', 'OZ'],
        'OZ':  ['PO3', 'PO4', 'O2', 'O1'],
        'O2':  ['PO4', 'OZ'],
    }

    def __init__(self,
                 root_path: str = './Processed_data',
                 chunk_size: int = 250,
                 overlap: int = 0,
                 num_channel: int = 30,
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
        file_name = os.path.basename(
            record)  # an element from file name list, such as 'sub087.pkl'
        # derive the given arguments (kwargs)
        with open(os.path.join(root_path, file_name), 'rb') as f:
            samples = pkl.load(
                f, encoding='iso-8859-1'
            )  # 28(trials), 32(channels), 30s*250hz(time points)
        return {
            'samples': samples
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 28
        num_channels = 32
        sampling_rate = 250
        duration = 30
        time_points = duration * sampling_rate

        samples = np.random.randn(num_trials, num_channels, time_points)

        return {
            'samples': samples,
            'record': 'sub087.pkl'
        }

    def process_record(self, record: str, samples: np.ndarray, chunk_size: int = None, overlap: int = None, num_channel: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        file_name = os.path.basename(record)
        subject_id = int(file_name.split('.')[0][3:])
        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, :num_channel]
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id,
                               'valence': _VALENCE_DICT[trial_id + 1], 'emotion': _EMOTION_DICT[trial_id + 1]}
            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size
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
        file_path_list = os.listdir(root_path)
        file_path_list.sort()
        file_path_list = [
            file_path for file_path in file_path_list if file_path.endswith('.pkl')
        ]
        return file_path_list


class FACEDDataset(CachedDataset):

    r"""
    The FACED dataset was provided by the Tsinghua Laboratory of Brain and Intelligence. The Finer-grained Affective Computing EEG Dataset (FACED) recorded 32-channel EEG signals from 123 subjects. During the experiment, subjects watched 28 emotion-elicitation video clips covering nine emotion categories (amusement, inspiration, joy, tenderness; anger, fear, disgust, sadness, and neutral emotion), providing a fine-grained and balanced categorization on both the positive and negative sides of emotion.
    This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Please refer to the downloaded URL.
    - Year: 2023
    - Download URL: https://www.synapse.org/#!Synapse:syn50614194/files/
    - Reference: Please refer to the downloaded URL.
    - Stimulus: video clips.
    - Signals: Electroencephalogram (30 channels at 250Hz) and two channels of left/right mastoid signals from 123 subjects.
    - Rating: 28 video clips are annotated in valence and discrete emotion dimensions. The valence is divided into positive (1), negative (-1), and neutral (0). Discrete emotions are divided into anger (0), disgust (1), fear (2), sadness (3), neutral (4), amusement (5), inspiration (6), joy (7), and tenderness (8).

    In order to use this dataset, the download folder :obj:`Processed_data` (download from this url: https://www.synapse.org/#!Synapse:syn50615881) is required, containing the following files:

    .. code-block:: text

        Processed_data/
        ├── sub000.pkl
        ├── sub001.pkl
        ├── sub002.pkl
        └── ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations


        def preprocess_fn(eeg, baseline=None, **kwargs) -> dict:
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(get_locations(
                    FACEDDataset.channels, FACEDDataset.grid)),
            ])
            return {
                'eeg': eeg_transform(eeg),
                'baseline': eeg_transform(baseline) if baseline is not None else None,
            }


        def transform_fn(eeg, baseline=None, **kwargs) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            return {
                'eeg': eeg_transform(eeg),
                'baseline': eeg_transform(baseline) if baseline is not None else None,
            }


        def label_fn(info: dict, **kwargs) -> dict:
            label_transform = transforms.Compose([
                transforms.Select('emotion'),
            ])
            return {
                'label': label_transform(info),
            }


        dataset = FACEDDataset(
            root_path='./Processed_data',
            offline_transform=preprocess_fn,
            online_transform=transform_fn,
            label_transform=label_fn,
        )

        print(dataset[0])

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDDataset
        from torcheeg import transforms


        def transform_fn(eeg, baseline=None, **kwargs) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.To2d(),
            ])
            return {
                'eeg': eeg_transform(eeg),
                'baseline': eeg_transform(baseline) if baseline is not None else None,
            }


        def label_fn(info: dict, **kwargs) -> dict:
            label_transform = transforms.Compose([
                transforms.Select('emotion'),
            ])
            return {
                'label': label_transform(info),
            }


        dataset = FACEDDataset(
            root_path='./Processed_data',
            online_transform=transform_fn,
            label_transform=label_fn,
        )

        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms import ToG


        def transform_fn(eeg, baseline=None, **kwargs) -> dict:
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(FACEDDataset.channels, FACEDDataset.adjacency)),
            ])
            return {
                'eeg': eeg_transform(eeg),
                'baseline': eeg_transform(baseline) if baseline is not None else None,
            }


        def label_fn(info: dict, **kwargs) -> dict:
            label_transform = transforms.Compose([
                transforms.Select('emotion'),
            ])
            return {
                'label': label_transform(info),
            }


        dataset = FACEDDataset(
            root_path='./Processed_data',
            online_transform=transform_fn,
            label_transform=label_fn,
        )

        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in matlab (unzipped Processed_data.zip) formats (default: :obj:`'./Processed_data'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 30 channels are EEG signals. (default: :obj:`30`)



        offline_transform (Callable, optional): EEG transform applied before storage; returns a dictionary containing eeg. (default: :obj:`None`)
        online_transform (Callable, optional): EEG transform applied on each read. (default: :obj:`None`)
        label_transform (Callable, optional): Transform info into a dictionary containing label on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = FACEDData

    channels = FACEDData.channels

    grid = FACEDData.grid

    adjacency = FACEDData.adjacency

    def __init__(self, root_path: str = './Processed_data', chunk_size: int = 250, overlap: int = 0, num_channel: int = 30, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, label_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel,
                  'offline_transform': offline_transform, 'online_transform': online_transform, 'label_transform': label_transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'label_transform': self.label_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
