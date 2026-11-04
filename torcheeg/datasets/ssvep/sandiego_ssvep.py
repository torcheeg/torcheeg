import os
import re
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.io import loadmat

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class SanDiegoSSVEPData(BaseData):
    """Read and preprocess records for :class:`SanDiegoSSVEPDataset`."""

    channels = [
        'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1',  'OZ',  'O2',
    ]

    grid = [
        ['PO7', 'PO3', 'POZ', 'PO4', 'PO8'],
        ['-',   'O1',  'OZ',  'O2',  '-'],
    ]

    adjacency = {
        'O1':  ['PO7', 'PO3', 'POZ', 'OZ'],
        'O2':  ['POZ', 'PO4', 'PO8', 'OZ'],
        'OZ':  ['PO3', 'POZ', 'PO4', 'O1', 'O2'],
        'PO3': ['PO7', 'POZ', 'O1', 'OZ'],
        'PO4': ['POZ', 'PO8', 'OZ', 'O2'],
        'PO7': ['PO3', 'O1'],
        'PO8': ['PO4', 'O2'],
        'POZ': ['PO3', 'PO4', 'O1', 'OZ', 'O2'],
    }

    frequencies = [9.25, 11.25, 13.25, 9.75,
                   11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

    def __init__(self,
                 root_path='./archive',
                 chunk_size: int = 256,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size
        )

    def read_record(self,
                    record: str,
                    root_path: str = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        file_path = os.path.join(root_path, record)
        data = loadmat(file_path)
        eeg = data['X'].transpose(-1, -2, -3)
        y = data['y'][0]

        return {
            'eeg': eeg,
            'y': y,
        }

    @staticmethod
    def fake_record(record: str,
                    root_path: str = './archive', **kwargs) -> Dict:
        n_trials = 10
        n_channels = 8
        n_times = 1000

        eeg = np.random.randn(n_trials, n_channels, n_times)
        y = np.random.randint(1, 5, size=(n_trials,))

        return {
            'record': 'S01trainEEG.mat',
            'eeg': eeg,
            'y': y
        }

    def process_record(self, record: str, eeg: np.ndarray, y: np.ndarray, chunk_size: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        subject_id = int(re.findall('S(\\d+).*\\.mat', record)[0])
        train = True if re.findall('S\\d+(train)EEG\\.mat', record) else False
        record_global_info = {'subject_id': subject_id, 'train': train}
        for trial_id, eeg_clip in enumerate(eeg):
            eeg_clip = eeg_clip[:, :chunk_size]
            label = int(y[trial_id] - 1)
            trial_id = f'train_{trial_id}' if train else f'test_{trial_id}'
            record_info = {'label': label, 'trial_id': trial_id}
            record_info.update(record_global_info)
            yield {'eeg': eeg_clip, 'info': record_info}

    def set_records(self, root_path=None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        file_names = os.listdir(root_path)
        return file_names


class SanDiegoSSVEPDataset(CachedDataset):

    r"""
    San Diego Square Joint Frequnecy-Phase Modulation SSVEP Dataset: lightweight dataset for studying SSVEP. For more information, please refer to the details below.


    - Author: Masaki Nakanishi et al.
    - Year: 2015
    - Download URL: https://www.kaggle.com/datasets/lzyuuu/ssvep-sandiego
    - Reference: Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,"A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10, e140703, 2015.
    - Stimulus: 12 different frequencies and phases of visual stimuli.
    - Signals: Electroencephalogram (8 channels at 256Hz). Training and testing sets have been divided for each participant (totally 10 participants) in original datasets .
    - label: The order of the stimulus frequencies in the EEG data is [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz which are labeled to range(0,12).

    In order to use this dataset, the download folder is required, containing the following files:

    .. code-block:: text

        archive/
        ├── S01testEEG.mat
        ├── S01trainEEG.mat
        ├── S02testEEG.mat
        ├── ...
        ├── S010testEEG.mat
        └── S010trainEEG.mat

    An example:

    .. code-block:: python

        from torcheeg.transforms import Select, BandSignal


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                BandSignal(sampling_rate=256, band_dict={'frequency_range': [8, 16]}),
            ])
            label_transform = transforms.Compose([
                Select('label'),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = SanDiegoSSVEPDataset(
            root_path='./archive',
            chunk_size=512,
            io_path='./tmp_out/sandiego',
            online_transform=transform_fn,
        )
        print(dataset[0]['eeg'].shape)  # EEG shape(1,8,512)
        print(dataset[0]['label'])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./archive'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`256`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    frequencies = SanDiegoSSVEPData.frequencies

    data_class = SanDiegoSSVEPData

    channels = SanDiegoSSVEPData.channels

    grid = SanDiegoSSVEPData.grid

    adjacency = SanDiegoSSVEPData.adjacency

    def __init__(self, root_path='./archive', chunk_size: int = 256, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path,
                  'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'io_path': self.io_path, 'io_size': self.io_size, 'io_mode': self.io_mode, 'num_worker': self.num_worker, 'verbose': self.verbose})
