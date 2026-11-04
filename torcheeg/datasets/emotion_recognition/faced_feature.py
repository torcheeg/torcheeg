import os
import pickle as pkl
from typing import Callable, Dict, Tuple, Union

import numpy as np

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset
from .faced import _EMOTION_DICT, _VALENCE_DICT


class FACEDFeatureData(BaseData):
    """Read and preprocess records for :class:`FACEDFeatureDataset`."""

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
                 root_path: str = './EEG_Features/DE',
                 num_channel: int = 30,
                 transform: Union[None, Callable] = None):

        self._rename_pkl_files(root_path)

        super().__init__(
            transform=transform,
            root_path=root_path,
            num_channel=num_channel
        )

    def _rename_pkl_files(self, root_path):
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith('.pkl.pkl'):
                    old_file_path = os.path.join(dirpath, filename)
                    # new file name, remove the last repetitive '.pkl'
                    new_filename = filename[:-4]
                    new_file_path = os.path.join(dirpath, new_filename)
                    # rename file
                    os.rename(old_file_path, new_file_path)

    def read_record(self, record: str, **kwargs) -> Dict:
        with open(os.path.join(record), 'rb') as f:
            data = pkl.load(
                f, encoding='iso-8859-1'
            )  # 28(trials), 32(channels), 30s(time points), 5(frequency bands)
        return {
            'data': data
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_trials = 28
        num_channels = 32
        time_points = 30
        freq_bands = 5

        data = np.random.rand(num_trials, num_channels,
                              time_points, freq_bands)
        data = np.abs(data)
        return {
            'data': data,
            'record': 'sub087.pkl'
        }

    def process_record(self, record: str, data: np.ndarray, num_channel: int = None, **kwargs):
        if num_channel is None:
            num_channel = self.num_channel
        file_name = os.path.basename(record)
        subject_id = int(file_name.split('.')[0][3:])
        for trial_id in range(len(data)):
            trial_samples = data[trial_id, :num_channel]
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id}
            for i in range(trial_samples.shape[1]):
                t_eeg = trial_samples[:, i]
                record_info = {'start_at': i * 250, 'end_at': (
                    i + 1) * 250, 'valence': _VALENCE_DICT[trial_id + 1], 'emotion': _EMOTION_DICT[trial_id + 1]}
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'info': record_info}

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
            os.path.join(root_path, file_path) for file_path in file_path_list
            if file_path.endswith('.pkl')
        ]

        return file_path_list


class FACEDFeatureDataset(CachedDataset):

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

    In order to use this dataset, the download folder :obj:`EEG_Features` (download from this url: https://www.synapse.org/#!Synapse:syn52368847) is required, containing the following files:

    .. code-block:: text

        EEG_Features/
        ├── DE/
        │   ├── sub000.pkl.pkl  # will be renamed to sub000.pkl automatically
        │   ├── sub001.pkl.pkl
        │   ├── sub002.pkl.pkl
        │   └── ...
        └── PSD/
            ├── sub000.pkl.pkl
            └── ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets import FACEDDataset
        from torcheeg.datasets.utils import get_locations


        def preprocess_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToGrid(get_locations(FACEDDataset.channels, FACEDDataset.grid)),
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
                transforms.Select('emotion'),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = FACEDFeatureDataset(
            root_path='./EEG_Features/DE',
            offline_transform=preprocess_fn,
            online_transform=transform_fn,
        )
        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets import FACEDDataset
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.graph import ToG


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(FACEDDataset.channels, FACEDDataset.adjacency)),
            ])
            label_transform = transforms.Compose([
                transforms.Select('emotion'),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = FACEDFeatureDataset(
            root_path='./EEG_Features/DE',
            online_transform=transform_fn,
        )
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in matlab (unzipped EEG_Features.zip) formats (default: :obj:`'./EEG_Features/DE'`, optional: :obj:`'./EEG_Features/PSD'`)
        num_channel (int): Number of channels used, of which the first 30 channels are EEG signals. (default: :obj:`30`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = FACEDFeatureData

    channels = FACEDFeatureData.channels

    grid = FACEDFeatureData.grid

    adjacency = FACEDFeatureData.adjacency

    _rename_pkl_files = FACEDFeatureData._rename_pkl_files

    def __init__(self, root_path: str = './EEG_Features/DE', num_channel: int = 30, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'num_channel': num_channel, 'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path,
                  'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'num_channel': self.num_channel, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
